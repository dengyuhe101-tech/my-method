/*
	Copyright (C) 2026-... David Cornu
	for the Convolutional Interactive Artificial Neural Networks by/for Astrophysicists (CIANNA) Code

	Licensed under the Apache License, Version 2.0.
*/

#include "prototypes.h"

static chgate_param *cg_param;

static void chgate_prev_dim(layer *previous, int *dim)
{
	if(previous == NULL)
	{
		printf("\n ERROR: channel_gate can not be the first layer.\n\n");
		exit(EXIT_FAILURE);
	}

	switch(previous->type)
	{
		case DENSE:
			get_dense_output_dim(previous, dim);
			break;
		case CONV:
			get_conv_output_dim(previous, dim);
			break;
		case POOL:
			get_pool_output_dim(previous, dim);
			break;
		case NORM:
			get_norm_output_dim(previous, dim);
			break;
		case LRN:
			get_lrn_output_dim(previous, dim);
			break;
		case CHGATE:
			get_chgate_output_dim(previous, dim);
			break;
		default:
			printf("\n ERROR: unsupported previous layer for channel_gate.\n\n");
			exit(EXIT_FAILURE);
	}
}

static inline float chgate_scale(float gate)
{
	return 2.0f / (1.0f + expf(-gate));
}

static inline float chgate_scale_deriv(float gate)
{
	float s = 1.0f / (1.0f + expf(-gate));
	return 2.0f * s * (1.0f - s);
}

void naiv_forward_chgate_layer(layer *current)
{
	int c, b, p;
	network *net = current->c_network;
	cg_param = (chgate_param*)current->param;
	size_t flat_map = (size_t)cg_param->map_size[0] * cg_param->map_size[1] * cg_param->map_size[2];
	size_t map_batch = flat_map * net->batch_size;
	float *input = (float*)current->input;
	float *output = (float*)current->output;
	float *gate = (float*)cg_param->gate;

	if(current->previous != NULL)
		current->input = current->previous->output;
	input = (float*)current->input;

	#pragma omp parallel for private(b, p) schedule(guided, 2)
	for(c = 0; c < cg_param->nb_maps; c++)
	{
		float scale = chgate_scale(gate[c]);
		size_t c_off = (size_t)c * map_batch;
		for(b = 0; b < net->batch_size; b++)
		{
			size_t b_off = c_off + (size_t)b * flat_map;
			for(p = 0; p < (int)flat_map; p++)
				output[b_off + p] = input[b_off + p] * scale;
		}
	}

	current->activation(current);
}

void naiv_backward_chgate_layer(layer *current)
{
	int c, b, p;
	network *net = current->c_network;
	cg_param = (chgate_param*)current->param;
	size_t flat_map = (size_t)cg_param->map_size[0] * cg_param->map_size[1] * cg_param->map_size[2];
	size_t map_batch = flat_map * net->batch_size;
	float *input = (float*)current->input;
	float *delta = (float*)current->delta_o;
	float *prev_delta = (float*)current->previous->delta_o;
	float *gate = (float*)cg_param->gate;
	float *update = (float*)cg_param->update;

	#pragma omp parallel for private(b, p) schedule(guided, 2)
	for(c = 0; c < cg_param->nb_maps; c++)
	{
		float scale = chgate_scale(gate[c]);
		float scale_deriv = chgate_scale_deriv(gate[c]);
		double grad_gate = 0.0;
		size_t c_off = (size_t)c * map_batch;
		for(b = 0; b < net->batch_size; b++)
		{
			size_t b_off = c_off + (size_t)b * flat_map;
			for(p = 0; p < (int)flat_map; p++)
			{
				float d = delta[b_off + p];
				float x = input[b_off + p];
				prev_delta[b_off + p] = d * scale;
				grad_gate += (double)d * (double)x;
			}
		}
		if(!current->frozen)
		{
			update[c] = net->learning_rate / net->batch_size * (float)(grad_gate * scale_deriv)
				+ net->momentum * update[c];
			update[c] += net->learning_rate * net->weight_decay * gate[c];
			gate[c] -= update[c];
		}
	}

	current->previous->deriv_activation(current->previous);
}

int chgate_create(network *net, layer *previous, const char *activation, FILE *f_load, int f_bin)
{
	int i;
	int prev_dim[4] = {1, 1, 1, 1};
	size_t flat_map, total_size;
	long long int mem_approx = 0;
	layer *current = (layer*)malloc(sizeof(layer));

	net->net_layers[net->nb_layers] = current;
	current->c_network = net;
	net->nb_layers++;

	printf("L:%d - CREATING CHANNEL GATE ATTENTION LAYER ...\n", net->nb_layers);

	cg_param = (chgate_param*)malloc(sizeof(chgate_param));
	current->type = CHGATE;
	load_activation_type(current, activation);
	current->frozen = 0;
	current->previous = previous;
	current->dropout_rate = 0.0f;
	current->bias_value = 0.0f;
	current->input = previous->output;

	chgate_prev_dim(previous, prev_dim);
	cg_param->map_size = (int*)calloc(3, sizeof(int));
	for(i = 0; i < 3; i++)
		cg_param->map_size[i] = prev_dim[i];
	cg_param->nb_maps = prev_dim[3];

	flat_map = (size_t)cg_param->map_size[0] * cg_param->map_size[1] * cg_param->map_size[2];
	total_size = flat_map * cg_param->nb_maps * net->batch_size;

	current->output = (float*)calloc(total_size, sizeof(float));
	mem_approx += total_size * sizeof(float);
	cg_param->gate = (float*)calloc(cg_param->nb_maps, sizeof(float));
	mem_approx += cg_param->nb_maps * sizeof(float);

	if(!net->inference_only)
	{
		current->delta_o = (float*)calloc(total_size, sizeof(float));
		cg_param->update = (float*)calloc(cg_param->nb_maps, sizeof(float));
		cg_param->grad = NULL;
		mem_approx += total_size * sizeof(float) + cg_param->nb_maps * sizeof(float);
	}
	else
	{
		current->delta_o = NULL;
		cg_param->update = NULL;
		cg_param->grad = NULL;
	}

	if(f_load != NULL)
	{
		if(f_bin)
			fread(cg_param->gate, sizeof(float), cg_param->nb_maps, f_load);
		else
			for(i = 0; i < cg_param->nb_maps; i++)
				fscanf(f_load, "%f", &((float*)cg_param->gate)[i]);
	}

	current->nb_params = cg_param->nb_maps;
	current->param = cg_param;
	set_linear_param(current, (int)total_size, (int)total_size, (int)total_size, 0);

	switch(net->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_chgate_define(current);
			mem_approx = cuda_convert_chgate_layer(current);
			cuda_define_activation(current);
			#endif
			break;
		case C_BLAS:
		case C_NAIV:
		default:
			current->forward = naiv_forward_chgate_layer;
			current->backprop = naiv_backward_chgate_layer;
			define_activation(current);
			break;
	}

	printf("      Input/Output: %dx%dx%dx%d, trainable channel gates: %d, init scale: 1.00\n",
		cg_param->map_size[0], cg_param->map_size[1], cg_param->map_size[2], cg_param->nb_maps, cg_param->nb_maps);
	printf("      Approx layer RAM/VRAM requirement: %d MB\n", (int)(mem_approx / 1000000));
	net->total_nb_param += cg_param->nb_maps;
	net->memory_footprint += mem_approx;

	return net->nb_layers - 1;
}

void chgate_save(FILE *f, layer *current, int f_bin)
{
	int i;
	void *host_gate = NULL;
	char layer_type = 'A';
	cg_param = (chgate_param*)current->param;

	if(f_bin)
	{
		fwrite(&layer_type, sizeof(char), 1, f);
		fwrite(&cg_param->nb_maps, sizeof(int), 1, f);
		fwrite(cg_param->map_size, sizeof(int), 3, f);
		print_activ_param(f, current, f_bin);
	}
	else
	{
		fprintf(f, "A%dm%dx%dx%d", cg_param->nb_maps, cg_param->map_size[0], cg_param->map_size[1], cg_param->map_size[2]);
		print_activ_param(f, current, f_bin);
		fprintf(f, "\n");
	}

	if(current->c_network->compute_method == C_CUDA)
	{
		#ifdef CUDA
		host_gate = (float*)calloc(cg_param->nb_maps, sizeof(float));
		cuda_get_table_FP32(cg_param->FP32_gate, host_gate, cg_param->nb_maps);
		#endif
	}
	else
		host_gate = cg_param->gate;

	if(f_bin)
		fwrite(host_gate, sizeof(float), cg_param->nb_maps, f);
	else
	{
		for(i = 0; i < cg_param->nb_maps; i++)
			fprintf(f, "%g ", ((float*)host_gate)[i]);
		fprintf(f, "\n");
	}

	if(current->c_network->compute_method == C_CUDA && host_gate != NULL)
		free(host_gate);
}

void chgate_load(network *net, FILE *f, int f_bin, int skip_layer)
{
	int i, nb_maps;
	int map_size[3];
	char activ[40];
	layer *previous = NULL;

	if(f_bin)
	{
		fread(&nb_maps, sizeof(int), 1, f);
		fread(map_size, sizeof(int), 3, f);
		fread(activ, sizeof(char), 40, f);
	}
	else
	{
		fscanf(f, "%dm%dx%dx%d%s", &nb_maps, &map_size[0], &map_size[1], &map_size[2], activ);
		fscanf(f, "\n");
	}

	if(skip_layer)
	{
		float dummy;
		for(i = 0; i < nb_maps; i++)
		{
			if(f_bin)
				fread(&dummy, sizeof(float), 1, f);
			else
				fscanf(f, "%f", &dummy);
		}
		return;
	}

	if(net->nb_layers > 0)
		previous = net->net_layers[net->nb_layers - 1];
	chgate_create(net, previous, activ, f, f_bin);
}

void get_chgate_output_dim(layer *current, int *dim)
{
	cg_param = (chgate_param*)current->param;
	dim[0] = cg_param->map_size[0];
	dim[1] = cg_param->map_size[1];
	dim[2] = cg_param->map_size[2];
	dim[3] = cg_param->nb_maps;
}

void free_chgate(layer *current)
{
	cg_param = (chgate_param*)current->param;
	free(current->activ_param);
	#ifdef CUDA
	if(current->c_network->compute_method == C_CUDA)
		cuda_free_chgate(current);
	else
	#endif
	{
		free(current->output);
		free(cg_param->gate);
		if(!current->c_network->inference_only)
		{
			free(current->delta_o);
			free(cg_param->update);
		}
	}
	free(cg_param->map_size);
	free(cg_param);
	free(current);
}
