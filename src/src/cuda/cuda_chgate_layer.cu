/*
	Copyright (C) 2026-... David Cornu
	for the Convolutional Interactive Artificial Neural Networks by/for Astrophysicists (CIANNA) Code

	Licensed under the Apache License, Version 2.0.
*/

#include "../prototypes.h"

static int cu_blocks;
static chgate_param *cg_param;

__device__ __forceinline__ float chgate_sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__global__ void cuda_chgate_forward_kernel(float *output, const float *input, const float *gate,
	int nb_maps, int flat_map, int batch_size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t total = (size_t)nb_maps * flat_map * batch_size;
	if(i >= total)
		return;
	int c = i / ((size_t)flat_map * batch_size);
	float s = chgate_sigmoid(gate[c]);
	output[i] = input[i] * (2.0f * s);
}

__global__ void cuda_chgate_backward_kernel(float *prev_delta, const float *input, const float *delta,
	const float *gate, float *grad, int nb_maps, int flat_map, int batch_size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t map_batch = (size_t)flat_map * batch_size;
	size_t total = (size_t)nb_maps * map_batch;
	if(i >= total)
		return;
	int c = i / map_batch;
	float s = chgate_sigmoid(gate[c]);
	prev_delta[i] = delta[i] * (2.0f * s);
	atomicAdd(&grad[c], delta[i] * input[i] * (2.0f * s * (1.0f - s)));
}

__global__ void cuda_chgate_update_kernel(float *gate, float *update, const float *grad,
	float learning_rate, float momentum, float weight_decay, int batch_size, int nb_maps)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if(c >= nb_maps)
		return;
	update[c] = learning_rate / batch_size * grad[c] + momentum * update[c];
	update[c] += learning_rate * weight_decay * gate[c];
	gate[c] -= update[c];
}

size_t cuda_convert_chgate_layer(layer *current)
{
	size_t vram_approx = 0;
	network *net = current->c_network;
	cg_param = (chgate_param*)current->param;
	size_t flat_map = (size_t)cg_param->map_size[0] * cg_param->map_size[1] * cg_param->map_size[2];
	size_t total = flat_map * cg_param->nb_maps * net->batch_size;

	if(net->cu_inst.use_cuda_TC != FP32C_FP32A && net->cu_inst.use_cuda_TC != TF32C_FP32A)
	{
		printf("\n ERROR: channel_gate currently supports FP32C_FP32A/TF32C_FP32A CUDA storage only.\n\n");
		exit(EXIT_FAILURE);
	}

	vram_approx += cuda_convert_table(net, &(current->output), total, 0);
	vram_approx += cuda_convert_table_FP32(&(cg_param->gate), cg_param->nb_maps, 0);
	cg_param->FP32_gate = cg_param->gate;

	if(!net->inference_only)
	{
		vram_approx += cuda_convert_table(net, &(current->delta_o), total, 0);
		vram_approx += cuda_convert_table_FP32(&(cg_param->update), cg_param->nb_maps, 0);
		cudaMalloc(&(cg_param->grad), cg_param->nb_maps * sizeof(float));
		vram_approx += cg_param->nb_maps * sizeof(float);
	}

	return vram_approx;
}

void cuda_free_chgate(layer *current)
{
	cg_param = (chgate_param*)current->param;
	cudaFree(current->output);
	cudaFree(cg_param->gate);
	if(!current->c_network->inference_only)
	{
		cudaFree(current->delta_o);
		cudaFree(cg_param->update);
		cudaFree(cg_param->grad);
	}
}

void cuda_forward_chgate_layer(layer *current)
{
	network *net = current->c_network;
	if(net->length == 0)
		return;
	cg_param = (chgate_param*)current->param;
	size_t flat_map = (size_t)cg_param->map_size[0] * cg_param->map_size[1] * cg_param->map_size[2];
	size_t total = flat_map * cg_param->nb_maps * net->batch_size;
	if(current->previous != NULL)
		current->input = current->previous->output;
	cu_blocks = (total + cu_threads - 1) / cu_threads;
	cuda_chgate_forward_kernel<<<cu_blocks, cu_threads>>>((float*)current->output, (float*)current->input,
		(float*)cg_param->gate, cg_param->nb_maps, (int)flat_map, net->batch_size);
	current->activation(current);
}

void cuda_backward_chgate_layer(layer *current)
{
	network *net = current->c_network;
	cg_param = (chgate_param*)current->param;
	size_t flat_map = (size_t)cg_param->map_size[0] * cg_param->map_size[1] * cg_param->map_size[2];
	size_t total = flat_map * cg_param->nb_maps * net->batch_size;

	cudaMemset(cg_param->grad, 0, cg_param->nb_maps * sizeof(float));
	cu_blocks = (total + cu_threads - 1) / cu_threads;
	cuda_chgate_backward_kernel<<<cu_blocks, cu_threads>>>((float*)current->previous->delta_o, (float*)current->input,
		(float*)current->delta_o, (float*)cg_param->gate, (float*)cg_param->grad,
		cg_param->nb_maps, (int)flat_map, net->batch_size);

	if(!current->frozen)
	{
		cu_blocks = (cg_param->nb_maps + cu_threads - 1) / cu_threads;
		cuda_chgate_update_kernel<<<cu_blocks, cu_threads>>>((float*)cg_param->gate, (float*)cg_param->update,
			(float*)cg_param->grad, net->learning_rate, net->momentum, net->weight_decay,
			net->batch_size, cg_param->nb_maps);
	}

	current->previous->deriv_activation(current->previous);
}

void cuda_chgate_define(layer *current)
{
	current->forward = cuda_forward_chgate_layer;
	current->backprop = cuda_backward_chgate_layer;
}
