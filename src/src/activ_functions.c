
/*
	Copyright (C) 2026-... David Cornu
	for the Convolutional Interactive Artificial 
	Neural Networks by/for Astrophysicists (CIANNA) Code
	(https://github.com/Deyht/CIANNA)

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
*/


#include "prototypes.h"

// Public are in "prototypes.h"

// Private prototypes
void deriv_output_error(layer *current);
void output_error_fct(layer *current);

void linear_activation(layer *current);
void linear_deriv(layer *previous);
void linear_deriv_output_error(layer *current);
void linear_output_error(layer *current);

void fill_string_relu_activ_param(layer *current, char *activ);
void ReLU_activation(layer *current);
void ReLU_deriv(layer *previous);
void ReLU_deriv_output_error(layer *current);
void ReLU_output_error(layer *current);

void fill_string_logistic_activ_param(layer *current, char *activ);
void logistic_activation(layer *current);
void logistic_deriv(layer *previous);
void logistic_deriv_output_error(layer *current);
void logistic_output_error(layer *current);

void softmax_activation(layer *current);
void softmax_deriv(layer *previous);
void softmax_deriv_output_error(layer *current);
void softmax_output_error(layer *current);

float IoU_fct(float *output, float *target);
float GIoU_fct(float *output, float *target);
float DIoU_fct(float *output, float *target);
float DIoU2_fct(float *output, float *target);
float RotatedIoU_fct(float *output, float *target, float output_theta, float target_theta);

void YOLO_activation(layer *current);
void YOLO_deriv(layer *previous);
void YOLO_deriv_output_error(layer *current);
void YOLO_output_error(layer *current);

void linear_activation_fct(void *tab, int dim, int biased_dim, int offset, int length, size_t size);
void linear_deriv_fct(void *deriv, int dim, int biased_dim, int offset, int length, size_t size);
void ReLU_activation_fct(void *tab, int dim, int biased_dim, int offset, float saturation, float leaking_factor, int length, size_t size);
void ReLU_deriv_fct(void *deriv, void *value, int dim, int biased_dim,	int offset, float saturation, float leaking_factor, int length, size_t size);
void quadratic_deriv_output_error(void *delta_o, void *output, void *target, int dim, int biased_dim, int offset, int length, size_t size);
void quadratic_output_error(void *output_error, void *output, void *target, int dim, int biased_dim, int offset, int length, size_t size);
void logistic_activation_fct(void *tab, float beta, float saturation, int dim, int biased_dim, int offset, int length, size_t size);
void logistic_deriv_fct(void *deriv, void *value, float beta, int dim, int biased_dim, int offset, int length, size_t size);
void softmax_activation_fct(void *tab, int dim, int biased_dim, int offset, int length, int batch_size, size_t size);
void cross_entropy_deriv_output_error(void *delta_o, void *output, void *target, int dim, int biased_dim, int offset, int length, size_t size);
void cross_entropy_output_error(void *output_error, void *output, void *target, int dim, int biased_dim, int offset, int length, size_t size);
void YOLO_activation_fct(void *i_tab, int flat_offset, int len, yolo_param y_param, size_t size, int class_softmax);
void YOLO_deriv_error_fct(void *i_delta_o, void *i_output, void *i_target, int flat_target_size, int flat_output_size,
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size, int nb_im_iter);
void YOLO_error_fct(float *i_output_error, void *i_output, void *i_target, int flat_target_size, int flat_output_size,
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size);


void define_activation(layer *current)
{
	switch(current->activation_type)
	{
		case RELU:
			current->activation = ReLU_activation;
			current->deriv_activation = ReLU_deriv;
			break;
		
		case LOGISTIC:
			current->activation = logistic_activation;
			current->deriv_activation = logistic_deriv;
			break;
			
		case SOFTMAX:
			current->activation = softmax_activation;
			current->deriv_activation = softmax_deriv;
			break;
			
		case YOLO:
			current->activation = YOLO_activation;
			current->deriv_activation = YOLO_deriv;
			break;
			
		case LINEAR:
			default:
			current->activation = linear_activation;
			current->deriv_activation = linear_deriv;
			break;
	}

}


void deriv_output_error(layer *current)
{
	switch(current->activation_type)
	{
		case RELU:
			ReLU_deriv_output_error(current);
			break;
		
		case LOGISTIC:
			logistic_deriv_output_error(current);
			break;
			
		case SOFTMAX:
			softmax_deriv_output_error(current);
			break;
			
		case YOLO:
			YOLO_deriv_output_error(current);
			break;
			
		case LINEAR:
		default:
			linear_deriv_output_error(current);
			break;
	
	}
}


void output_error_fct(layer *current)
{
	switch(current->activation_type)
	{
		case RELU:
			ReLU_output_error(current);
			break;
		
		case LOGISTIC:
			logistic_output_error(current);
			break;
			
		case SOFTMAX:
			softmax_output_error(current);
			break;
		
		case YOLO:
			YOLO_output_error(current);
			break;
		
		case LINEAR:
		default:
			linear_output_error(current);
			break;
	}
}


void output_error(layer *current)
{
	switch(current->c_network->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_output_error_fct(current);
			#endif
			break;
		
		case C_NAIV:
		case C_BLAS:
		default:
			output_error_fct(current);
			break;
	}	
}


void output_deriv_error(layer *current)
{
	switch(current->c_network->compute_method)
	{
		case C_CUDA:
			#ifdef CUDA
			cuda_deriv_output_error(current);
			#endif
			break;
		
		case C_NAIV:
		case C_BLAS:
			deriv_output_error(current);
			break;
			
		default:
			deriv_output_error(current);
			break;
	}	
}


void fill_string_activ_param(layer *current, char *activ, int no_param)
{
	switch(current->activation_type)
	{
		case LOGISTIC:
			if(no_param)
				sprintf(activ,"LOGI");
			else
				fill_string_logistic_activ_param(current, activ);
			break;
		case SOFTMAX:
			sprintf(activ,"SMAX");
			break;
		case YOLO:
			sprintf(activ,"YOLO");
			break;
		case RELU:
			if(no_param)
				sprintf(activ,"RELU");
			else
				fill_string_relu_activ_param(current, activ);
			break;
		case LINEAR:
		default:
			sprintf(activ,"LIN");
			break;
	}
}


void print_activ_param(FILE *f, layer *current, int f_bin)
{
	char temp_string[40];

	fill_string_activ_param(current, temp_string, 0);
	
	if(f_bin)
		fwrite(temp_string, sizeof(char), 40, f);
	else
		fprintf(f, "%s ", temp_string);
}


void load_activation_type(layer *current, const char *activ)
{
	if(activ == NULL)
	{
		current->activation_type = LINEAR;
		return;
	}

	if(strncmp(activ, "SMAX", 4) == 0)
		current->activation_type = SOFTMAX;
	else if(strncmp(activ, "LIN", 3) == 0)
		current->activation_type = LINEAR;
	else if(strncmp(activ, "LOGI", 4) == 0)
		current->activation_type = LOGISTIC;
	else if(strncmp(activ, "YOLO", 4) == 0)
		current->activation_type = YOLO;
	else if(strncmp(activ, "RELU", 4) == 0)
		current->activation_type = RELU;
	else
		current->activation_type = LINEAR;
}


//#####################################################
//		 Linear activation related functions
//#####################################################

void set_linear_param(layer *current, int size, int dim, int biased_dim, int offset)
{
	current->activ_param = (linear_param*) malloc(sizeof(linear_param));
	linear_param *param = (linear_param*)current->activ_param;	
	
	param->size = size;
	param->dim = dim;
	param->biased_dim = biased_dim;
	param->offset = offset;
	current->bias_value = 0.5f;
}


void linear_activation_fct(void *tab, int dim, int biased_dim, int offset, int length, size_t size)
{
	size_t i;
	float *f_tab = (float*) tab;
	
	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i >= (length*biased_dim) && (i+1)%(dim+1) != 0)
				f_tab[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset >= length)
				f_tab[i] = 0.0f;
		}
	}
}


void linear_deriv_fct(void *deriv, int dim, int biased_dim, int offset, int length, size_t size)
{
	size_t i;
	float *f_deriv = (float*) deriv;
	
	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i >= (length*biased_dim) && (i+1)%(dim+1) != 0)
				f_deriv[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset >= length)
				f_deriv[i] = 0.0f;
		}
	}
}


void linear_activation(layer *current)
{
	linear_param *param = (linear_param*)current->activ_param;
	linear_activation_fct(current->output, param->dim, param->biased_dim, 
		param->offset, current->c_network->length, param->size);
}


void linear_deriv(layer *previous)
{
	linear_param *param = (linear_param*)previous->activ_param;
	linear_deriv_fct(previous->delta_o, param->dim, param->biased_dim, 
		param->offset, previous->c_network->length, param->size);
}


void linear_deriv_output_error(layer *current)
{
	linear_param *param = (linear_param*)current->activ_param;
	quadratic_deriv_output_error(current->delta_o, current->output, current->c_network->target,
		param->dim, param->biased_dim, param->offset, current->c_network->length, param->size);
	linear_deriv_fct(current->delta_o, param->dim, param->biased_dim,
		param->offset, current->c_network->length, param->size);
}


void linear_output_error(layer *current)
{	
	linear_param *param = (linear_param*)current->activ_param;
	quadratic_output_error(current->c_network->output_error, current->output, current->c_network->target, 
		param->dim, param->biased_dim, param->offset, current->c_network->length, param->size);
}


//#####################################################



//#####################################################
//		 ReLU activation related functions
//#####################################################

void set_relu_param(layer *current, int size, int dim, int biased_dim, int offset, const char *activ)
{
	char *temp = NULL;

	current->activ_param = (ReLU_param*) malloc(sizeof(ReLU_param));
	ReLU_param *param = (ReLU_param*)current->activ_param;	
	
	param->size = size;
	param->dim = dim;
	param->biased_dim = biased_dim;
	param->offset = offset;
	param->saturation = 800.0f;
	param->leaking_factor = 0.05f;
	current->bias_value = 0.1f;
	
	temp = strstr(activ, "_S");
	if(temp != NULL)
		sscanf(temp, "_S%f", &param->saturation);
	temp = strstr(activ, "_L");
	if(temp != NULL)
		sscanf(temp, "_L%f", &param->leaking_factor);
	
}


void fill_string_relu_activ_param(layer *current, char *activ)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	sprintf(activ,"RELU_S%0.2f_L%0.2f", param->saturation, param->leaking_factor);
}


//Is in fact a leaky ReLU, to obtain true ReLU define leaking_factor to 0
void ReLU_activation_fct(void *tab, int dim, int biased_dim, int offset, 
	float saturation, float leaking_factor, int length, size_t size)
{
	size_t i;
	float *f_tab = (float*) tab;
	
	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)
			{
				if(f_tab[i] <= 0.0f)
					f_tab[i] *= leaking_factor;
				else if(f_tab[i] > saturation)
					f_tab[i] = saturation + (f_tab[i] - saturation)*leaking_factor;
			}
			else
				f_tab[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				if(f_tab[i] <= 0.0f)
					f_tab[i] *= leaking_factor;
				else if(f_tab[i] > saturation)
					f_tab[i] = saturation + (f_tab[i] - saturation)*(leaking_factor);
			}
			else
				f_tab[i] = 0.0f;
		}
	}
}


void ReLU_deriv_fct(void *deriv, void *value, int dim, int biased_dim,	int offset,	
	 float saturation, float leaking_factor, int length, size_t size)
{
	size_t i;
	float *f_deriv = (float*) deriv;
	float *f_value = (float*) value;
	
	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)
			{
				if(f_value[i] <= 0.0f)
					f_deriv[i] *= leaking_factor;
				else if(f_deriv[i] > saturation)
					f_deriv[i] *= leaking_factor;
			}
			else
				f_deriv[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				if(f_value[i] <= 0.0f)
					f_deriv[i] *= leaking_factor;
				else if(f_deriv[i] > saturation)
					f_deriv[i] *= leaking_factor;
			}
			else
				f_deriv[i] = 0.0f;
		}
	}
}


void ReLU_activation(layer *current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	ReLU_activation_fct(current->output, param->dim, param->biased_dim, param->offset, param->saturation, 
		param->leaking_factor, current->c_network->length, param->size);
}


void ReLU_deriv(layer *previous)
{
	ReLU_param *param = (ReLU_param*)previous->activ_param;
	ReLU_deriv_fct(previous->delta_o, previous->output, param->dim, param->biased_dim, param->offset, 
		param->saturation, param->leaking_factor, previous->c_network->length, param->size);
}


void ReLU_deriv_output_error(layer *current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	
	quadratic_deriv_output_error(current->delta_o, current->output, current->c_network->target, param->dim, 
		param->biased_dim, param->offset, current->c_network->length, param->size);
	ReLU_deriv_fct(current->delta_o, current->output, param->dim, param->biased_dim,
		param->offset, param->saturation, param->leaking_factor, current->c_network->length, param->size);
}


void ReLU_output_error(layer *current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	
	quadratic_output_error(current->c_network->output_error, current->output, current->c_network->target, 
		param->dim, param->biased_dim, param->offset, current->c_network->length, param->size);
}


void quadratic_deriv_output_error(void *delta_o, void *output, void *target, int dim, 
	int biased_dim, int offset, int length, size_t size)
{
	size_t i;
	int nb_filters, c_batch, c_filter, in_filter_pos, pos;
	
	float *f_delta_o = (float*) delta_o;
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	int dim_offset = dim * offset;
	nb_filters = size / dim_offset;
	
	#pragma omp parallel for private(pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{	
		if(biased_dim > dim)
		{
			if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)
			{
				pos = i - i/(dim+1);
				f_delta_o[i] = (f_output[i] - f_target[pos]);
			}
			else
				f_delta_o[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				c_filter = i / dim_offset;
				c_batch = (i / dim)%offset;
				in_filter_pos = i % dim;
				
				pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;
				f_delta_o[i] = (f_output[i] - f_target[pos]);
			}
			else
				f_delta_o[i] = 0.0f;
		}
	}
}


void quadratic_output_error(void *output_error, void *output, void *target, int dim, 
	int biased_dim, int offset, int length, size_t size)
{
	size_t i;
	int nb_filters, c_batch, c_filter, in_filter_pos, pos;
	
	float *f_output_error = (float*) output_error;
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	int dim_offset = dim * offset;
	nb_filters = size / dim_offset;
	
	#pragma omp parallel for private(pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)
			{
				pos = i - i/(dim+1);
				f_output_error[i] = 0.5*(f_output[i] - f_target[pos])*(f_output[i] - f_target[pos]);
			}
			else
				f_output_error[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				c_filter = i / dim_offset;
				c_batch = (i / dim)%offset;
				in_filter_pos = i % dim;
				
				pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;
				f_output_error[i] = 0.5*(f_output[i] - f_target[pos])*(f_output[i] - f_target[pos]);
			}
			else
				f_output_error[i] = 0.0f;
		}
	}
}

//#####################################################


//#####################################################
//		 Logistic activation related functions
//#####################################################


void set_logistic_param(layer *current, int size, int dim, int biased_dim, int offset, const char *activ)
{
	char *temp = NULL;

	current->activ_param = (logistic_param*) malloc(sizeof(logistic_param));
	logistic_param *param = (logistic_param*)current->activ_param;	
	
	param->size = size;
	param->dim = dim;
	param->biased_dim = biased_dim;
	param->offset = offset;
	param->saturation = 6.0f;
	param->beta = 1.0f;
	current->bias_value = -1.0f;
	
	temp = strstr(activ, "_S");
	if(temp != NULL)
		sscanf(temp, "_S%f", &param->saturation);
	temp = strstr(activ, "_B");
	if(temp != NULL)
		sscanf(temp, "_B%f", &param->beta);
	
}


void fill_string_logistic_activ_param(layer *current, char *activ)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	sprintf(activ,"LOGI_S%0.2f_B%0.2f", param->saturation, param->beta);
}


void logistic_activation(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	logistic_activation_fct(current->output, param->beta, param->saturation, param->dim, 
		param->biased_dim, param->offset, current->c_network->length, param->size);
}


void logistic_activation_fct(void *tab, float beta, float saturation, int dim,
	int biased_dim, int offset, int length, size_t size)
{
	size_t i = 0;
	
	float *f_tab = (float*) tab;

	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)
			{
				f_tab[i] = -beta*f_tab[i];
				if(f_tab[i] > saturation)
					f_tab[i] = saturation;
				f_tab[i] = 1.0f/(1.0f + expf(f_tab[i]));
			}
			else
				f_tab[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				f_tab[i] = -beta*f_tab[i];
				if(f_tab[i] > saturation)
					f_tab[i] = saturation;
				f_tab[i] = 1.0f/(1.0f + expf(f_tab[i]));
			}
			else
				f_tab[i] = 0.0f;
		}
	}
}


void logistic_deriv(layer *previous)
{
	logistic_param *param = (logistic_param*)previous->activ_param;
	logistic_deriv_fct(previous->delta_o, previous->output, param->beta, param->dim, 
		param->biased_dim, param->offset, previous->c_network->length, param->size);
}


void logistic_deriv_fct(void *deriv, void *value, float beta, int dim,
	int biased_dim, int offset, int length, size_t size)
{
	size_t i;
	
	float *f_deriv = (float*) deriv;
	float *f_value = (float*) value;
	
	#pragma omp parallel for schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)
				f_deriv[i] *= beta*f_value[i]*(1.0-f_value[i]);
			else
				f_deriv[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
				f_deriv[i] *= beta*f_value[i]*(1.0-f_value[i]);
			else
				f_deriv[i] = 0.0f;
		}
	}
}


void logistic_deriv_output_error(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	quadratic_deriv_output_error(current->delta_o, current->output, current->c_network->target, param->dim, 
		param->biased_dim, param->offset, current->c_network->length, param->size);
	logistic_deriv_fct(current->delta_o, current->output, param->beta, param->dim, 
		param->biased_dim, param->offset, current->c_network->length, param->size);
}


void logistic_output_error(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	quadratic_output_error(current->c_network->output_error, current->output, current->c_network->target,
		param->dim, param->biased_dim, param->offset, current->c_network->length, param->size);
}

//#####################################################



//#####################################################
//		 Soft-Max activation related functions
//#####################################################


void set_softmax_param(layer *current, int size, int dim, int biased_dim, int offset)
{
	current->activ_param = (softmax_param*) malloc(sizeof(softmax_param));
	softmax_param *param = (softmax_param*)current->activ_param;	
	
	param->size = size;
	param->dim = dim;
	param->biased_dim = biased_dim;
	param->offset = offset;
	current->bias_value = 0.1f;
}


void softmax_activation(layer *current)
{
	softmax_param *param = (softmax_param*)current->activ_param;
	softmax_activation_fct(current->output, param->dim, param->biased_dim, param->offset, 
		current->c_network->length, current->c_network->batch_size, param->size);
}


void softmax_activation_fct(void *tab, int dim, int biased_dim,
	int offset, int length, int batch_size, size_t size)
{
	int i, j, k, l;
	float *pos, *off_pos;
	float vmax;
	float normal = 0.0f;
	int batched_dim = dim * batch_size;
	int nb_filters = size / batched_dim;
	
	#pragma omp parallel for private(j, k, l, pos, off_pos, vmax, normal) schedule(guided,4)
	for(i = 0; i < batch_size; i++)
	{
		pos = (float*)tab + i*biased_dim;
		normal = 0.0f;
		
		if(biased_dim > dim)
		{
			if(i < length)
			{
				vmax = *pos;
				for(j = 0; j < dim; j++)
				{
					off_pos = pos + j*offset;
					if(*off_pos > vmax)
						vmax = *off_pos;
				}
				
				for(j = 0; j < dim; j++)
				{
					off_pos = pos + j*offset;
					*off_pos = expf(*off_pos-vmax);
					normal += *off_pos;
				}
				pos[dim*offset] = 0.0f;
				
				for(j = 0; j < dim; j++)
				{
					off_pos = pos + j*offset;
					*off_pos /= normal;
				}
				pos[dim*offset] = 0.0f;
			}
			else
			{
				for(j = 0; j < dim; j++)
				{
					off_pos = pos + j*offset;
					*off_pos = 0.0f;
				}
				pos[dim*offset] = 0.0f;
			}
		}
		else
		{
			if(i < length)
			{
				vmax = *pos;
				for(k = 0; k < nb_filters ; k++)
				{
					for(l = 0; l < dim; l++)
					{
						off_pos = pos + k*batched_dim + l;
						if(*off_pos > vmax)
							vmax = *off_pos;
					}
				}
				
				for(k = 0; k < nb_filters ; k++)
				{
					for(l = 0; l < dim; l++)
					{
						off_pos = pos + k*batched_dim + l;
						*off_pos = expf((*off_pos-vmax));
						normal += *off_pos;
					}
				}
				
				for(k = 0; k < nb_filters ; k++)
				{
					for(l = 0; l < dim; l++)
					{
						off_pos = pos + k*batched_dim + l;
						*off_pos /= normal;
					}
				}
				}
			else
			{
				for(k = 0; k < nb_filters ; k++)
				{
					for(l = 0; l < dim; l++)
					{
						off_pos = pos + k*batched_dim + l;
						*off_pos = 0.0f;
					}
				}
			}
		}
	}
}


void cross_entropy_deriv_output_error(void *delta_o, void *output, void *target, 
	int dim, int biased_dim, int offset, int length, size_t size)
{
	size_t i;
	int nb_filters, c_batch, c_filter, in_filter_pos, pos;
	
	float *f_delta_o = (float*) delta_o; 
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	int length_biased_dim = length * biased_dim;
	int dim_offset = dim * offset;
	nb_filters = size / dim_offset;
	
	#pragma omp parallel for private(c_batch, c_filter, in_filter_pos, pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length_biased_dim) && (i+1)%(dim+1) != 0)
			{
				pos = i - i/(dim+1);
				f_delta_o[i] = (f_output[i] - f_target[pos]);
			}
			else
				f_delta_o[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				c_filter = i / dim_offset;
				c_batch = (i / dim)%offset;
				in_filter_pos = i % dim;
				
				pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;
				f_delta_o[i] = (f_output[i] - f_target[pos]);
			}
			else
				f_delta_o[i] = 0.0f;
		}
	}
}


void cross_entropy_output_error(void *output_error, void *output, void *target, 
	int dim, int biased_dim, int offset, int length, size_t size)
{
	size_t i;
	int nb_filters, c_batch, c_filter, in_filter_pos, pos;
	
	float *f_output_error = (float*) output_error;
	float *f_output = (float*) output;
	float *f_target = (float*) target;
	
	int length_biased_dim = length * biased_dim;
	int dim_offset = dim * offset;
	nb_filters = size / dim_offset;
	
	#pragma omp parallel for private(c_batch, c_filter, in_filter_pos, pos) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		if(biased_dim > dim)
		{
			if(i < (length_biased_dim) && (i+1)%(dim+1) != 0)
			{
				pos = i - i/(dim+1);
				if(f_output[i] > 0.000001f)
					f_output_error[i] = -f_target[pos]*logf(f_output[i]);
				else
					f_output_error[i] = -f_target[pos]*logf(0.000001f);
			}
			else
				f_output_error[i] = 0.0f;
		}
		else
		{
			if((i / dim)%offset < length)
			{
				c_filter = i / dim_offset;
				c_batch = (i / dim)%offset;
				in_filter_pos = i % dim;
				
				pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;
				if(f_output[i] > 0.000001f)
					f_output_error[i] = -f_target[pos]*logf(f_output[i]);
				else
					f_output_error[i] = -f_target[pos]*logf(0.000001f);
			}
			else
				f_output_error[i] = 0.0f;
		}
	}
}


void softmax_deriv(layer *previous)
{
	printf("ERROR : Softmax can not be used in the middle of the network !\n");
	exit(EXIT_FAILURE);
}


void softmax_deriv_output_error(layer *current)
{
	softmax_param *param = (softmax_param*)current->activ_param;
	cross_entropy_deriv_output_error(current->delta_o, current->output, current->c_network->target,
		param->dim, param->biased_dim, param->offset, current->c_network->length, param->size);
}


void softmax_output_error(layer *current)
{
	softmax_param *param = (softmax_param*)current->activ_param;
	cross_entropy_output_error(current->c_network->output_error, current->output, 
		current->c_network->target, param->dim, param->biased_dim, param->offset, 
		current->c_network->length, param->size);
}


//#####################################################


//#####################################################
//		 YOLO activation related functions
//#####################################################

void set_yolo_param(layer *current)
{
	int i, j;
	float *temp = NULL;
	
	current->activ_param = (yolo_param*) malloc(sizeof(yolo_param));
	yolo_param *param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	
	//From global YOLO settings
	yolo_param *global_param = (yolo_param*)current->c_network->y_param;
	
	//Content copy (and not only pointer) to keep network properties accessible
	//all necessary pointers are redifined in the following lines
	*param = *(current->c_network->y_param);
	
	int nb_box = param->nb_box;
	int nb_class = param->nb_class;
	int nb_param = global_param->nb_param;
	int nb_angle = global_param->nb_angle;
	int max_obj_per_image = param->max_nb_obj_per_image;
	int output_offset = 8+nb_class+nb_param+nb_angle;
	
	int nb_filters = c_param->nb_filters;
	int total_nb_area = c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2];
	int batched_total_nb_area = total_nb_area * current->c_network->batch_size;
	
	if(nb_box*output_offset != nb_filters)
	{
		printf("\n ERROR: Nb filters size mismatch in YOLO dimensions!\n");
		printf("%d %d\n", nb_box*output_offset, nb_filters);
		exit(EXIT_FAILURE);
	}
	
	temp = (float*) calloc(6*3, sizeof(float));
	param->slopes_and_maxes_tab = (float**) malloc(6*sizeof(float*));
	param->prior_size = (float*) calloc(nb_box*3, sizeof(float));
	
	/* Having prior relative to image size is a good idea in principle, but it hides the fact that the network has a given receptive field related to its architecture.
	Therefore, increasing the input resolution will end up to cases where object sizes are too large.
	For now we prefer to have priors as fixed pixel sizes, allowing to processing cutout of different sizes from a given input image (without rescale).*/
	for(i = 0; i < nb_box; i++)
		for(j = 0; j < 3; j++)
			param->prior_size[i*3+j] = fmax(1.0f, global_param->prior_size[i*3+j]);
	
	for(i = 0; i < 6; i++)
	{
		param->slopes_and_maxes_tab[i] = &temp[i*3];
		for(j = 0; j < 3; j++)
			  param->slopes_and_maxes_tab[i][j] = global_param->slopes_and_maxes_tab[i][j];
	}
	
	param->size = total_nb_area * c_param->nb_filters * current->c_network->batch_size;
	
	param->dim = param->size;
	param->biased_dim = param->dim;
	param->cell_size = (int*) calloc(3, sizeof(int));
	for (i = 0; i < 3; i++)
		param->cell_size[i] = current->c_network->in_dims[i] / c_param->nb_area[i];
	
	param->IoU_monitor = (float*) calloc(2 * nb_box * batched_total_nb_area, sizeof(float));
	param->target_cell_mask = (int*) calloc(batched_total_nb_area * max_obj_per_image, sizeof(int));
	param->IoU_table = (float*) calloc(batched_total_nb_area * max_obj_per_image * nb_box, sizeof(float));
	param->dist_prior = (float*) calloc(batched_total_nb_area * max_obj_per_image * nb_box, sizeof(float));
	param->box_locked = (int*) calloc(batched_total_nb_area * nb_box, sizeof(int));
	param->box_in_pix = (float*) calloc(batched_total_nb_area * 6 * nb_box, sizeof(float));
	
	current->bias_value = 0.5;
}


float IoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d;
	
	inter_w = fmaxf(0.0f, fminf(output[3], target[3]) - fmaxf(output[0], target[0]));
	inter_h = fmaxf(0.0f, fminf(output[4], target[4]) - fmaxf(output[1], target[1]));
	inter_d = fmaxf(0.0f, fminf(output[5], target[5]) - fmaxf(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d = fabs(output[3]-output[0])*fabs(output[4]-output[1])*fabs(output[5]-output[2])
			+ fabs(target[3]-target[0])*fabs(target[4]-target[1])*fabs(target[5]-target[2])
			- inter_3d;
	
	return ((float)inter_3d)/(float)uni_3d;
}


float GIoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_3d, enclose_w, enclose_h, enclose_d;
	
	inter_w = fmaxf(0.0f, fminf(output[3], target[3]) - fmaxf(output[0], target[0]));
	inter_h = fmaxf(0.0f, fminf(output[4], target[4]) - fmaxf(output[1], target[1]));
	inter_d = fmaxf(0.0f, fminf(output[5], target[5]) - fmaxf(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d = fabs(output[3]-output[0])*fabs(output[4]-output[1])*fabs(output[5]-output[2])
			+ fabs(target[3]-target[0])*fabs(target[4]-target[1])*fabs(target[5]-target[2])
			- inter_3d;
	enclose_w = (fmaxf(output[3], target[3]) - fminf(output[0], target[0]));
	enclose_h = (fmaxf(output[4], target[4]) - fminf(output[1], target[1]));
	enclose_d = (fmaxf(output[5], target[5]) - fminf(output[2], target[2]));
	enclose_3d = enclose_w * enclose_h * enclose_d;
	
	return (((float)inter_3d)/(float)uni_3d - (float)(enclose_3d - uni_3d)/(float)enclose_3d);
}


//order: xmin, ymin, zmin, xmax, ymax, zmax
float DIoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_w, enclose_h, enclose_d;
	float cx_a, cx_b, cy_a, cy_b, cz_a, cz_b, dist_cent, diag_enclose;
	
	inter_w = fmaxf(0.0f, fminf(output[3], target[3]) - fmaxf(output[0], target[0]));
	inter_h = fmaxf(0.0f, fminf(output[4], target[4]) - fmaxf(output[1], target[1]));
	inter_d = fmaxf(0.0f, fminf(output[5], target[5]) - fmaxf(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d = fabs(output[3]-output[0])*fabs(output[4]-output[1])*fabs(output[5]-output[2])
			+ fabs(target[3]-target[0])*fabs(target[4]-target[1])*fabs(target[5]-target[2])
			- inter_3d;
	enclose_w = (fmaxf(output[3], target[3]) - fminf(output[0], target[0]));
	enclose_h = (fmaxf(output[4], target[4]) - fminf(output[1], target[1]));
	enclose_d = (fmaxf(output[5], target[5]) - fminf(output[2], target[2]));
	
	cx_a = (output[3] + output[0])*0.5; cx_b = (target[3] + target[0])*0.5; 
	cy_a = (output[4] + output[1])*0.5; cy_b = (target[4] + target[1])*0.5;
	cz_a = (output[5] + output[2])*0.5; cz_b = (target[5] + target[2])*0.5;
	dist_cent = sqrt((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b) + (cz_a - cz_b)*(cz_a - cz_b));
	diag_enclose = sqrt(enclose_w*enclose_w + enclose_h*enclose_h + enclose_d*enclose_d);
	
	return ((float)inter_3d)/(float)uni_3d - (float)(dist_cent/diag_enclose);
}


//order: xmin, ymin, zmin, xmax, ymax, zmax
float DIoU2_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_w, enclose_h, enclose_d;
	float cx_a, cx_b, cy_a, cy_b, cz_a, cz_b, dist_cent, diag_enclose;
	
	inter_w = fmaxf(0.0f, fminf(output[3], target[3]) - fmaxf(output[0], target[0]));
	inter_h = fmaxf(0.0f, fminf(output[4], target[4]) - fmaxf(output[1], target[1]));
	inter_d = fmaxf(0.0f, fminf(output[5], target[5]) - fmaxf(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d = fabs(output[3]-output[0])*fabs(output[4]-output[1])*fabs(output[5]-output[2])
			+ fabs(target[3]-target[0])*fabs(target[4]-target[1])*fabs(target[5]-target[2]) - inter_3d;
			
	enclose_w = (fmaxf(output[3], target[3]) - fminf(output[0], target[0]));
	enclose_h = (fmaxf(output[4], target[4]) - fminf(output[1], target[1]));
	enclose_d = (fmaxf(output[5], target[5]) - fminf(output[2], target[2]));
	
	cx_a = (output[3] + output[0])*0.5; cx_b = (target[3] + target[0])*0.5; 
	cy_a = (output[4] + output[1])*0.5; cy_b = (target[4] + target[1])*0.5;
	cz_a = (output[5] + output[2])*0.5; cz_b = (target[5] + target[2])*0.5;
	dist_cent = ((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b) + (cz_a - cz_b)*(cz_a - cz_b));
	diag_enclose = (enclose_w*enclose_w + enclose_h*enclose_h + enclose_d*enclose_d);
	
	return ((float)inter_3d)/(float)uni_3d - (float)(dist_cent/diag_enclose);
}

static float yolo_polygon_signed_area(const float *poly, int n)
{
	float area = 0.0f;
	for(int i = 0; i < n; i++)
	{
		int j = (i + 1) % n;
		area += poly[2*i] * poly[2*j+1] - poly[2*i+1] * poly[2*j];
	}
	return 0.5f * area;
}

static float yolo_polygon_area(const float *poly, int n)
{
	return fabsf(yolo_polygon_signed_area(poly, n));
}

static int yolo_inside_half_plane(const float *point, const float *edge_start, const float *edge_end, int clip_ccw)
{
	float edge_x = edge_end[0] - edge_start[0];
	float edge_y = edge_end[1] - edge_start[1];
	float rel_x = point[0] - edge_start[0];
	float rel_y = point[1] - edge_start[1];
	float cross = edge_x * rel_y - edge_y * rel_x;
	return clip_ccw ? (cross >= -1.0e-6f) : (cross <= 1.0e-6f);
}

static void yolo_line_intersection(const float *p1, const float *p2, const float *q1, const float *q2, float *out)
{
	float r_x = p2[0] - p1[0];
	float r_y = p2[1] - p1[1];
	float s_x = q2[0] - q1[0];
	float s_y = q2[1] - q1[1];
	float denom = r_x * s_y - r_y * s_x;
	if(fabsf(denom) < 1.0e-8f)
	{
		out[0] = p2[0];
		out[1] = p2[1];
		return;
	}
	float t = ((q1[0] - p1[0]) * s_y - (q1[1] - p1[1]) * s_x) / denom;
	out[0] = p1[0] + t * r_x;
	out[1] = p1[1] + t * r_y;
}

static int yolo_polygon_clip(const float *subject, int subject_n, const float *clip, int clip_n, float *result)
{
	float input[32], output[32], inter[2];
	int output_n = subject_n;
	int clip_ccw = yolo_polygon_signed_area(clip, clip_n) >= 0.0f;
	for(int i = 0; i < 2*subject_n; i++)
		output[i] = subject[i];

	for(int edge_i = 0; edge_i < clip_n; edge_i++)
	{
		int input_n = output_n;
		for(int i = 0; i < 2*input_n; i++)
			input[i] = output[i];
		output_n = 0;
		if(input_n <= 0)
			break;

		const float *edge_start = clip + 2*edge_i;
		const float *edge_end = clip + 2*((edge_i + 1) % clip_n);
		float prev[2] = {input[2*(input_n-1)], input[2*(input_n-1)+1]};
		int prev_inside = yolo_inside_half_plane(prev, edge_start, edge_end, clip_ccw);

		for(int i = 0; i < input_n; i++)
		{
			float curr[2] = {input[2*i], input[2*i+1]};
			int curr_inside = yolo_inside_half_plane(curr, edge_start, edge_end, clip_ccw);
			if(curr_inside)
			{
				if(!prev_inside && output_n < 16)
				{
					yolo_line_intersection(prev, curr, edge_start, edge_end, inter);
					output[2*output_n] = inter[0];
					output[2*output_n+1] = inter[1];
					output_n++;
				}
				if(output_n < 16)
				{
					output[2*output_n] = curr[0];
					output[2*output_n+1] = curr[1];
					output_n++;
				}
			}
			else if(prev_inside && output_n < 16)
			{
				yolo_line_intersection(prev, curr, edge_start, edge_end, inter);
				output[2*output_n] = inter[0];
				output[2*output_n+1] = inter[1];
				output_n++;
			}
			prev[0] = curr[0];
			prev[1] = curr[1];
			prev_inside = curr_inside;
		}
	}

	for(int i = 0; i < 2*output_n; i++)
		result[i] = output[i];
	return output_n;
}

static void yolo_obb_corners(float cx, float cy, float w, float h, float theta, float *poly)
{
	if(h > w)
	{
		float tmp = w;
		w = h;
		h = tmp;
		theta += 1.5707963267948966f;
	}
	w = fmaxf(w, 1.0e-6f);
	h = fmaxf(h, 1.0e-6f);
	float c = cosf(theta);
	float s = sinf(theta);
	float local[8] = {-0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f};
	for(int i = 0; i < 4; i++)
	{
		float u = local[2*i] * w;
		float v = local[2*i+1] * h;
		poly[2*i] = cx + u*c + v*s;
		poly[2*i+1] = cy - u*s + v*c;
	}
}

float RotatedIoU_fct(float *output, float *target, float output_theta, float target_theta)
{
	float out_poly[8], targ_poly[8], inter_poly[32];
	float output_w = fabsf(output[3] - output[0]);
	float output_h = fabsf(output[4] - output[1]);
	float target_w = fabsf(target[3] - target[0]);
	float target_h = fabsf(target[4] - target[1]);
	float output_cx = 0.5f * (output[0] + output[3]);
	float output_cy = 0.5f * (output[1] + output[4]);
	float target_cx = 0.5f * (target[0] + target[3]);
	float target_cy = 0.5f * (target[1] + target[4]);

	yolo_obb_corners(output_cx, output_cy, output_w, output_h, output_theta, out_poly);
	yolo_obb_corners(target_cx, target_cy, target_w, target_h, target_theta, targ_poly);
	int inter_n = yolo_polygon_clip(out_poly, 4, targ_poly, 4, inter_poly);
	float inter_area = (inter_n <= 0) ? 0.0f : yolo_polygon_area(inter_poly, inter_n);
	float output_area = fmaxf(output_w * output_h, 1.0e-6f);
	float target_area = fmaxf(target_w * target_h, 1.0e-6f);
	float uni = output_area + target_area - inter_area;
	if(uni <= 1.0e-6f)
		return 0.0f;
	return fminf(1.0f, fmaxf(0.0f, inter_area / uni));
}

static float yolo_box_quality(yolo_param *y_param, float *out_int, float *targ_int, float *output, float *target,
	int l_o, int l_t, size_t f_offset, int nb_class, int nb_param, int nb_angle)
{
	if(y_param->IoU_type != ROTIOU || nb_angle < 2)
		return y_param->c_IoU_fct(out_int, targ_int);

	float out_cos = (float)output[(l_o+8+nb_class+nb_param+0)*f_offset];
	float out_sin = (float)output[(l_o+8+nb_class+nb_param+1)*f_offset];
	float targ_cos = (float)target[l_t+7+nb_param+0];
	float targ_sin = (float)target[l_t+7+nb_param+1];
	float out_theta = 0.5f * atan2f(out_sin, out_cos);
	float targ_theta = 0.5f * atan2f(targ_sin, targ_cos);
	return RotatedIoU_fct(out_int, targ_int, out_theta, targ_theta);
}

static void yolo_load_target_box(float *target, int l_t, float *targ_int, int target_box_mode)
{
	if(target_box_mode > 0)
	{
		for(int l = 0; l < 3; l++)
		{
			float c = target[l_t+1+l];
			float s = fmaxf(target[l_t+4+l], 1.0e-6f);
			targ_int[l] = c - 0.5f*s;
			targ_int[l+3] = c + 0.5f*s;
		}
	}
	else
	{
		for(int l = 0; l < 6; l++)
			targ_int[l] = target[l_t+1+l];
	}
}

static float yolo_obb_cov_loss_terms(
	float pred_cx, float pred_cy, float pred_w, float pred_h, float pred_theta,
	float targ_cx, float targ_cy, float targ_w, float targ_h, float targ_theta,
	float *grad_logw, float *grad_logh, float *grad_theta)
{
	pred_w = fmaxf(pred_w, 1.0e-6f);
	pred_h = fmaxf(pred_h, 1.0e-6f);
	targ_w = fmaxf(targ_w, 1.0e-6f);
	targ_h = fmaxf(targ_h, 1.0e-6f);
	float center_norm = fmaxf(targ_w*targ_h, 1.0f);
	float cov_norm = fmaxf((targ_w*targ_w + targ_h*targ_h)*(targ_w*targ_w + targ_h*targ_h), 1.0f);
	float dcx = pred_cx - targ_cx;
	float dcy = pred_cy - targ_cy;
	float center_loss = (dcx*dcx + dcy*dcy) / center_norm;

	float cp = cosf(pred_theta), sp = sinf(pred_theta);
	float ct = cosf(targ_theta), st = sinf(targ_theta);
	float ap = 0.25f*pred_w*pred_w, bp = 0.25f*pred_h*pred_h;
	float at = 0.25f*targ_w*targ_w, bt = 0.25f*targ_h*targ_h;

	float p_xx = ap*cp*cp + bp*sp*sp;
	float p_yy = ap*sp*sp + bp*cp*cp;
	float p_xy = (bp-ap)*sp*cp;
	float t_xx = at*ct*ct + bt*st*st;
	float t_yy = at*st*st + bt*ct*ct;
	float t_xy = (bt-at)*st*ct;

	float e_xx = p_xx - t_xx;
	float e_yy = p_yy - t_yy;
	float e_xy = p_xy - t_xy;
	float cov_loss = (e_xx*e_xx + e_yy*e_yy + 2.0f*e_xy*e_xy) / cov_norm;

	float dxx_logw = 2.0f*ap*cp*cp;
	float dyy_logw = 2.0f*ap*sp*sp;
	float dxy_logw = -2.0f*ap*sp*cp;
	float dxx_logh = 2.0f*bp*sp*sp;
	float dyy_logh = 2.0f*bp*cp*cp;
	float dxy_logh = 2.0f*bp*sp*cp;
	float dxx_theta = 2.0f*(bp-ap)*sp*cp;
	float dyy_theta = 2.0f*(ap-bp)*sp*cp;
	float dxy_theta = (bp-ap)*(cp*cp - sp*sp);

	*grad_logw = (e_xx*dxx_logw + e_yy*dyy_logw + 2.0f*e_xy*dxy_logw) / cov_norm;
	*grad_logh = (e_xx*dxx_logh + e_yy*dyy_logh + 2.0f*e_xy*dxy_logh) / cov_norm;
	*grad_theta = (e_xx*dxx_theta + e_yy*dyy_theta + 2.0f*e_xy*dxy_theta) / cov_norm;
	return center_loss + cov_loss;
}

static float yolo_clamp_quality(float value, float floor)
{
	if(value < floor)
		return floor;
	if(value > 0.98f)
		return 0.98f;
	return value;
}

static int yolo_target_is_da_like(yolo_param *y_param, float *target, int l_t)
{
	float flux = (float)target[l_t+7+0];
	float w = fabsf((float)target[l_t+4]);
	float h = fabsf((float)target[l_t+5]);
	float aspect;

	if(y_param->obj_quality_da_flux_max >= 0.0f && flux > y_param->obj_quality_da_flux_max)
		return 0;

	if(y_param->obj_quality_da_bmaj_log_max > y_param->obj_quality_da_bmaj_log_min &&
		y_param->obj_quality_da_bmin_log_max > y_param->obj_quality_da_bmin_log_min)
	{
		float bmaj_log = (float)target[l_t+7+1] *
			(y_param->obj_quality_da_bmaj_log_max - y_param->obj_quality_da_bmaj_log_min) +
			y_param->obj_quality_da_bmaj_log_min;
		float bmin_log = (float)target[l_t+7+2] *
			(y_param->obj_quality_da_bmin_log_max - y_param->obj_quality_da_bmin_log_min) +
			y_param->obj_quality_da_bmin_log_min;
		aspect = expf(bmaj_log - bmin_log);
	}
	else
	{
		if(w <= 1.0e-6f || h <= 1.0e-6f)
			return 0;
		aspect = w / h;
		if(aspect < 1.0f)
			aspect = 1.0f / aspect;
	}
	if(y_param->obj_quality_da_aspect_min > 0.0f && aspect < y_param->obj_quality_da_aspect_min)
		return 0;
	if(y_param->obj_quality_da_aspect_max > 0.0f && aspect > y_param->obj_quality_da_aspect_max)
		return 0;
	return 1;
}

static float yolo_probability_quality_target(yolo_param *y_param, float *output, float *target,
	int l_o, int l_t, size_t f_offset, int nb_class, int nb_param, int nb_angle, float max_IoU, float *obj_in_offset)
{
	float floor = y_param->prob_quality_floor;
	float q_box = yolo_clamp_quality((1.0f + max_IoU) * 0.5f, floor);
	float err_sum = 0.0f, weight_sum = 0.0f;
	float q_phys, target_quality;
	float center_err2 = 0.0f;
	int k, dim_count = y_param->fit_dim;

	if(y_param->prob_quality_mode <= 0)
		return 0.98f;
	if(y_param->prob_quality_mode == 1)
		return q_box;
	if(y_param->prob_quality_mode == 3)
	{
		if(dim_count > 3)
			dim_count = 3;
		if(dim_count <= 0)
			return q_box;
		for(k = 0; k < dim_count; k++)
		{
			float diff = (float)output[(l_o+k)*f_offset] - obj_in_offset[k];
			center_err2 += diff * diff;
		}
		return yolo_clamp_quality(1.0f - sqrtf(center_err2 / (float)dim_count), floor);
	}

	if(nb_param > 0)
	{
		float diff = fabsf((float)output[(l_o+8+nb_class+0)*f_offset] - (float)target[l_t+7+0]);
		err_sum += 0.5f * diff;
		weight_sum += 0.5f;
	}
	if(nb_param > 3)
	{
		float diff = fabsf((float)output[(l_o+8+nb_class+3)*f_offset] - (float)target[l_t+7+3]);
		err_sum += diff;
		weight_sum += 1.0f;
	}
	if(nb_param > 4)
	{
		float diff = fabsf((float)output[(l_o+8+nb_class+4)*f_offset] - (float)target[l_t+7+4]);
		err_sum += diff;
		weight_sum += 1.0f;
	}
	if(nb_angle >= 4)
	{
		float out0 = (float)output[(l_o+8+nb_class+nb_param+2)*f_offset];
		float out1 = (float)output[(l_o+8+nb_class+nb_param+3)*f_offset];
		float targ0 = (float)target[l_t+7+nb_param+2];
		float targ1 = (float)target[l_t+7+nb_param+3];
		float out_norm = sqrtf(out0*out0 + out1*out1);
		float targ_norm = sqrtf(targ0*targ0 + targ1*targ1);
		float pa_quality = 0.5f;
		float angle_weight = (float)target[l_t+7+nb_param+nb_angle];
		float pa_weight = fminf(1.0f, fmaxf(0.0f, angle_weight * 0.5f));
		if(out_norm > 1.0e-6f && targ_norm > 1.0e-6f)
			pa_quality = 0.5f * (1.0f + (out0*targ0 + out1*targ1) / (out_norm*targ_norm));
		pa_quality = fminf(1.0f, fmaxf(0.0f, pa_quality));
		err_sum += pa_weight * (1.0f - pa_quality);
		weight_sum += pa_weight;
	}

	if(weight_sum <= 1.0e-6f)
		return q_box;
	q_phys = expf(-y_param->prob_quality_scale * err_sum / weight_sum);
	target_quality = q_box * q_phys;
	return yolo_clamp_quality(target_quality, floor);
}

static float yolo_objectness_quality_target(yolo_param *y_param, float *output, float *target,
	int l_o, int l_t, size_t f_offset, int nb_class, int nb_param, float max_IoU, float *obj_in_offset)
{
	float floor = y_param->obj_quality_floor;
	float q_geom = yolo_clamp_quality((1.0f + max_IoU) * 0.5f, floor);
	float q_center = q_geom;
	float q_phys = q_geom;
	float center_err2 = 0.0f, phys_err = 0.0f;
	float center_weight, geom_weight, phys_weight, weight_sum, quality;
	int k, dim_count = y_param->fit_dim;

	if(y_param->obj_quality_mode <= 0)
		return q_geom;

	if(dim_count > 3)
		dim_count = 3;
	if(dim_count > 0)
	{
		for(k = 0; k < dim_count; k++)
		{
			float diff = (float)output[(l_o+k)*f_offset] - obj_in_offset[k];
			center_err2 += diff * diff;
		}
		q_center = 1.0f - sqrtf(center_err2 / (float)dim_count);
		q_center = yolo_clamp_quality(q_center, floor);
	}

	if(nb_param > 0)
	{
		for(k = 0; k < nb_param; k++)
			phys_err += fabsf((float)output[(l_o+8+nb_class+k)*f_offset] - (float)target[l_t+7+k]);
		q_phys = expf(-y_param->obj_quality_scale * phys_err / (float)nb_param);
		q_phys = yolo_clamp_quality(q_phys, floor);
	}

	center_weight = y_param->obj_quality_center_weight;
	geom_weight = y_param->obj_quality_geom_weight;
	phys_weight = y_param->obj_quality_phys_weight;
	if(y_param->obj_quality_mode == 2 && yolo_target_is_da_like(y_param, target, l_t))
	{
		center_weight = y_param->obj_quality_da_center_weight;
		geom_weight = y_param->obj_quality_da_geom_weight;
		phys_weight = y_param->obj_quality_da_phys_weight;
	}

	weight_sum = center_weight + geom_weight + phys_weight;
	if(weight_sum <= 1.0e-6f)
		return q_geom;
	quality = (
		center_weight * q_center
		+ geom_weight * q_geom
		+ phys_weight * q_phys
	) / weight_sum;
	return yolo_clamp_quality(quality, floor);
}

static float yolo_smooth_l1_grad(float x)
{
	if(x > 1.0f)
		return 1.0f;
	if(x < -1.0f)
		return -1.0f;
	return x;
}

static void yolo_add_scorer_aux_delta(yolo_param *y_param, float *delta_o, float *output, float *target,
	int l_o, int l_t, size_t f_offset, int nb_class, int nb_param, float *obj_in_offset,
	int *cell_size, float **sm_tab, float coord_scale, float param_scale, float *param_ind_scale,
	float max_IoU, float min_param_IoU_lim, int diff_flag, int targ_diff_flag)
{
	float sample_weight, bmaj_log_t, bmin_log_t, bmaj_t, bmin_t, conv_arcsec, conv_pix;
	float flux_range, bmaj_range, bmin_range;
	float err, grad, outv, scale;
	int k, dim_count;

	if(y_param->scorer_aux_mode <= 0 || nb_param < 3)
		return;
	if(max_IoU <= min_param_IoU_lim || (diff_flag != 0 && targ_diff_flag >= 2))
		return;

	sample_weight = yolo_target_is_da_like(y_param, target, l_t) ? y_param->scorer_aux_da_weight : 1.0f;
	if(sample_weight <= 0.0f)
		sample_weight = 1.0f;

	bmaj_range = y_param->obj_quality_da_bmaj_log_max - y_param->obj_quality_da_bmaj_log_min;
	bmin_range = y_param->obj_quality_da_bmin_log_max - y_param->obj_quality_da_bmin_log_min;
	conv_pix = 8.0f;
	if(bmaj_range > 1.0e-6f && bmin_range > 1.0e-6f && y_param->scorer_aux_pixel_arcsec > 0.0f)
	{
		bmaj_log_t = (float)target[l_t+7+1] * bmaj_range + y_param->obj_quality_da_bmaj_log_min;
		bmin_log_t = (float)target[l_t+7+2] * bmin_range + y_param->obj_quality_da_bmin_log_min;
		bmaj_t = expf(bmaj_log_t);
		bmin_t = expf(bmin_log_t);
		conv_arcsec = sqrtf(fmaxf(bmaj_t, bmin_t) * fmaxf(bmaj_t, bmin_t)
			+ y_param->scorer_aux_beam_arcsec * y_param->scorer_aux_beam_arcsec);
		conv_pix = fmaxf(1.0f, conv_arcsec / y_param->scorer_aux_pixel_arcsec);
	}

	if(y_param->scorer_aux_center_scale > 0.0f)
	{
		dim_count = y_param->fit_dim;
		if(dim_count > 2)
			dim_count = 2;
		for(k = 0; k < dim_count; k++)
		{
			outv = (float)output[(l_o+k)*f_offset];
			scale = ((float)cell_size[k]) / conv_pix;
			err = (outv - obj_in_offset[k]) * scale;
			grad = yolo_smooth_l1_grad(err) * scale;
			delta_o[(l_o+k)*f_offset] += sample_weight * y_param->scorer_aux_center_scale
				* sm_tab[0][0] * coord_scale * outv * (1.0f - outv) * grad;
		}
	}

	if(y_param->scorer_aux_flux_scale > 0.0f)
	{
		flux_range = y_param->scorer_aux_flux_log_max - y_param->scorer_aux_flux_log_min;
		if(flux_range <= 1.0e-6f)
			flux_range = 1.0f;
		err = ((float)output[(l_o+8+nb_class+0)*f_offset] - (float)target[l_t+7+0]) * flux_range / 0.50f;
		grad = yolo_smooth_l1_grad(err) * flux_range / 0.50f;
		delta_o[(l_o+8+nb_class+0)*f_offset] += sample_weight * y_param->scorer_aux_flux_scale
			* param_ind_scale[0] * sm_tab[5][0] * param_scale * grad;
	}

	if(y_param->scorer_aux_size_scale > 0.0f && bmaj_range > 1.0e-6f && bmin_range > 1.0e-6f)
	{
		err = (((float)output[(l_o+8+nb_class+1)*f_offset] - (float)target[l_t+7+1]) * bmaj_range) / 0.35f;
		grad = yolo_smooth_l1_grad(err) * bmaj_range / 0.35f;
		delta_o[(l_o+8+nb_class+1)*f_offset] += sample_weight * y_param->scorer_aux_size_scale
			* param_ind_scale[1] * sm_tab[5][0] * param_scale * grad;

		err = (((float)output[(l_o+8+nb_class+2)*f_offset] - (float)target[l_t+7+2]) * bmin_range) / 0.35f;
		grad = yolo_smooth_l1_grad(err) * bmin_range / 0.35f;
		delta_o[(l_o+8+nb_class+2)*f_offset] += sample_weight * y_param->scorer_aux_size_scale
			* param_ind_scale[2] * sm_tab[5][0] * param_scale * grad;
	}
}

static int yolo_flux_refine_is_delta_param(yolo_param *y_param, int k, int nb_param)
{
	return (
		y_param->flux_refine_mode > 0
		&& nb_param >= 4
		&& y_param->flux_refine_delta_param_index >= 0
		&& y_param->flux_refine_delta_param_index < nb_param
		&& k == y_param->flux_refine_delta_param_index
	);
}

static int yolo_flux_refine_is_gate_param(yolo_param *y_param, int k, int nb_param)
{
	return (
		y_param->flux_refine_mode >= 2
		&& nb_param >= 5
		&& y_param->flux_refine_gate_param_index >= 0
		&& y_param->flux_refine_gate_param_index < nb_param
		&& k == y_param->flux_refine_gate_param_index
	);
}

static int yolo_flux_refine_is_aux_param(yolo_param *y_param, int k, int nb_param)
{
	return (
		yolo_flux_refine_is_delta_param(y_param, k, nb_param)
		|| yolo_flux_refine_is_gate_param(y_param, k, nb_param)
	);
}

static void yolo_add_flux_refine_delta(yolo_param *y_param, float *delta_o, float *output, float *target,
	int l_o, int l_t, size_t f_offset, int nb_class, int nb_param, float **sm_tab,
	float param_scale, float *param_ind_scale, float max_IoU, float min_param_IoU_lim,
	int diff_flag, int targ_diff_flag)
{
	int delta_k = y_param->flux_refine_delta_param_index;
	int gate_k = y_param->flux_refine_gate_param_index;
	float flux_range, base_norm, delta_raw, gate_raw, gate, final_norm, target_norm;
	float residual_norm, delta_norm, ungated_abs_err, base_abs_err, gate_target;
	float err, grad, delta_scale, gate_margin, d_gate;

	if(y_param->flux_refine_mode <= 0 || nb_param < 4)
		return;
	if(delta_k < 0 || delta_k >= nb_param)
		return;
	if(max_IoU <= min_param_IoU_lim || (diff_flag != 0 && targ_diff_flag >= 2))
		return;

	delta_scale = (y_param->flux_refine_delta_norm_scale > 0.0f) ? y_param->flux_refine_delta_norm_scale : 0.25f;
	flux_range = y_param->scorer_aux_flux_log_max - y_param->scorer_aux_flux_log_min;
	if(flux_range <= 1.0e-6f)
		flux_range = 1.0f;

	base_norm = (float)output[(l_o+8+nb_class+0)*f_offset];
	delta_raw = (float)output[(l_o+8+nb_class+delta_k)*f_offset];
	target_norm = (float)target[l_t+7+0];

	if(y_param->flux_refine_mode < 2)
	{
		if(y_param->flux_refine_loss_scale <= 0.0f)
			return;
		final_norm = base_norm + delta_scale * delta_raw;
		err = (final_norm - target_norm) * flux_range / 0.50f;
		grad = yolo_smooth_l1_grad(err) * flux_range / 0.50f * delta_scale;
		delta_o[(l_o+8+nb_class+delta_k)*f_offset] += y_param->flux_refine_loss_scale
			* param_ind_scale[delta_k] * sm_tab[5][0] * param_scale * grad;

		if(y_param->flux_refine_detach_base == 0)
		{
			grad = yolo_smooth_l1_grad(err) * flux_range / 0.50f;
			delta_o[(l_o+8+nb_class+0)*f_offset] += y_param->flux_refine_loss_scale
				* param_ind_scale[0] * sm_tab[5][0] * param_scale * grad;
		}
		return;
	}

	if(nb_param < 5 || gate_k < 0 || gate_k >= nb_param || gate_k == delta_k)
		return;

	gate_raw = (float)output[(l_o+8+nb_class+gate_k)*f_offset];
	gate = fminf(fmaxf(gate_raw, 0.0f), 1.0f);
	d_gate = (gate_raw >= 0.0f && gate_raw <= 1.0f) ? 1.0f : 0.0f;
	delta_norm = delta_scale * delta_raw;
	residual_norm = target_norm - base_norm;

	if(y_param->flux_refine_loss_scale > 0.0f)
	{
		err = (delta_norm - residual_norm) * flux_range / 0.50f;
		grad = yolo_smooth_l1_grad(err) * flux_range / 0.50f * delta_scale;
		delta_o[(l_o+8+nb_class+delta_k)*f_offset] += y_param->flux_refine_loss_scale
			* param_ind_scale[delta_k] * sm_tab[5][0] * param_scale * grad;
	}

	if(y_param->flux_refine_gate_loss_scale > 0.0f)
	{
		gate_margin = (y_param->flux_refine_gate_margin_norm > 0.0f) ? y_param->flux_refine_gate_margin_norm : 0.01f;
		base_abs_err = fabsf(residual_norm);
		ungated_abs_err = fabsf(delta_norm - residual_norm);
		gate_target = (ungated_abs_err + gate_margin < base_abs_err) ? 1.0f : 0.0f;
		grad = gate_raw - gate_target;
		delta_o[(l_o+8+nb_class+gate_k)*f_offset] += y_param->flux_refine_gate_loss_scale
			* param_ind_scale[gate_k] * sm_tab[5][0] * param_scale * grad;
	}

	if(y_param->flux_refine_final_loss_scale > 0.0f)
	{
		final_norm = base_norm + gate * delta_norm;
		err = (final_norm - target_norm) * flux_range / 0.50f;
		grad = yolo_smooth_l1_grad(err) * flux_range / 0.50f;
		delta_o[(l_o+8+nb_class+delta_k)*f_offset] += y_param->flux_refine_final_loss_scale
			* param_ind_scale[delta_k] * sm_tab[5][0] * param_scale * grad * gate * delta_scale;
		delta_o[(l_o+8+nb_class+gate_k)*f_offset] += y_param->flux_refine_final_loss_scale
			* param_ind_scale[gate_k] * sm_tab[5][0] * param_scale * grad * delta_norm * d_gate;
		if(y_param->flux_refine_detach_base == 0)
		{
			delta_o[(l_o+8+nb_class+0)*f_offset] += y_param->flux_refine_final_loss_scale
				* param_ind_scale[0] * sm_tab[5][0] * param_scale * grad;
		}
	}
}


int set_yolo_config(network *net, size_t nb_box, int nb_class, int nb_param, int max_nb_obj_per_image, const char *IoU_type_char, 
		const char *prior_dist_type_char, float *prior_size, float *yolo_noobj_prob_prior, int fit_dim, 
		int strict_box_size, int rand_startup, float rand_prob_best_box_assoc, float rand_prob, float min_prior_forced_scaling, float *scale_tab, 
		float **slopes_and_maxes_tab, float *param_ind_scale, float *IoU_limits, int *fit_parts, int class_softmax, 
		int diff_flag, const char *error_type, int no_override, int raw_output, int nb_angle, float angle_scale, 
		float angle_unit_norm_scale, float angle_slope, float angle_fmax, float angle_fmin, float min_angle_IoU_lim, int fit_angle,
	int angle_loss_mode, int target_box_mode, int obb_loss_mode, float obb_loss_scale,
	int multi_pos_topk, float multi_pos_iou_ratio, float multi_pos_min_iou, float multi_pos_obj_weight,
	int prob_quality_mode, float prob_quality_scale, float prob_quality_floor,
	int obj_quality_mode, float obj_quality_scale, float obj_quality_floor,
	float obj_quality_center_weight, float obj_quality_geom_weight, float obj_quality_phys_weight,
		float obj_quality_da_center_weight, float obj_quality_da_geom_weight, float obj_quality_da_phys_weight,
		float obj_quality_da_flux_max, float obj_quality_da_aspect_min, float obj_quality_da_aspect_max,
		float obj_quality_da_bmaj_log_max, float obj_quality_da_bmaj_log_min,
		float obj_quality_da_bmin_log_max, float obj_quality_da_bmin_log_min,
		float scorer_aux_flux_log_max, float scorer_aux_flux_log_min,
		int scorer_aux_mode, float scorer_aux_center_scale, float scorer_aux_flux_scale, float scorer_aux_size_scale,
		float scorer_aux_da_weight, float scorer_aux_pixel_arcsec, float scorer_aux_beam_arcsec,
		int flux_refine_mode, int flux_refine_delta_param_index, int flux_refine_gate_param_index,
		int flux_refine_detach_base, float flux_refine_loss_scale, float flux_refine_gate_loss_scale,
		float flux_refine_final_loss_scale, float flux_refine_delta_norm_scale, float flux_refine_gate_margin_norm)
{
	int i;
	int angle_target_dim = 0;
	float *temp;
	float **sm;
	float *l_IoU_limits, *l_scale_tab;
	float **l_slopes_and_maxes_tab;
	int *l_fit_parts;
	char display_IoU_type_char[40];
	char display_prior_dist_type[40];
	char display_error_type[40];
	char display_class_type[60];
	char display_difficult[40];
	
	if(net->y_param != NULL && net->y_param->fit_dim > 0)
	{
		printf("\n ERROR: Trying to update existing YOLO layer setup is not supported yet\n");
		exit(EXIT_FAILURE);
	}
	
	// Default setting
	net->y_param = (yolo_param*) malloc(sizeof(yolo_param));
	net->y_param->nb_box = 0;
	net->y_param->IoU_type = IOU;
	net->y_param->strict_box_size_association = 0;
	net->y_param->nb_class = 0;
	net->y_param->nb_param = 0;
		net->y_param->nb_angle = 0;
		net->y_param->angle_loss_mode = 0;
		net->y_param->target_box_mode = 0;
		net->y_param->obb_loss_mode = 0;
		net->y_param->obb_loss_scale = 0.0f;
		net->y_param->multi_pos_topk = 1;
		net->y_param->multi_pos_iou_ratio = 0.60f;
		net->y_param->multi_pos_min_iou = 0.05f;
		net->y_param->multi_pos_obj_weight = 0.35f;
		net->y_param->prob_quality_mode = 0;
		net->y_param->obj_quality_mode = 0;
	net->y_param->prob_quality_scale = 1.0f;
	net->y_param->prob_quality_floor = 0.02f;
	net->y_param->obj_quality_scale = 1.0f;
	net->y_param->obj_quality_floor = 0.02f;
	net->y_param->obj_quality_center_weight = 0.60f;
	net->y_param->obj_quality_geom_weight = 0.25f;
	net->y_param->obj_quality_phys_weight = 0.15f;
	net->y_param->obj_quality_da_center_weight = 0.65f;
	net->y_param->obj_quality_da_geom_weight = 0.35f;
	net->y_param->obj_quality_da_phys_weight = 0.0f;
	net->y_param->obj_quality_da_flux_max = 0.20f;
	net->y_param->obj_quality_da_aspect_min = 1.5f;
	net->y_param->obj_quality_da_aspect_max = 3.0f;
	net->y_param->obj_quality_da_bmaj_log_max = 0.0f;
		net->y_param->obj_quality_da_bmaj_log_min = 0.0f;
		net->y_param->obj_quality_da_bmin_log_max = 0.0f;
		net->y_param->obj_quality_da_bmin_log_min = 0.0f;
		net->y_param->scorer_aux_flux_log_max = 0.0f;
		net->y_param->scorer_aux_flux_log_min = 0.0f;
		net->y_param->scorer_aux_mode = 0;
		net->y_param->scorer_aux_center_scale = 0.0f;
		net->y_param->scorer_aux_flux_scale = 0.0f;
		net->y_param->scorer_aux_size_scale = 0.0f;
		net->y_param->scorer_aux_da_weight = 1.0f;
		net->y_param->scorer_aux_pixel_arcsec = 0.6042492f;
		net->y_param->scorer_aux_beam_arcsec = 0.625f;
		net->y_param->flux_refine_mode = 0;
		net->y_param->flux_refine_delta_param_index = 3;
		net->y_param->flux_refine_gate_param_index = 4;
		net->y_param->flux_refine_detach_base = 1;
		net->y_param->flux_refine_loss_scale = 0.0f;
		net->y_param->flux_refine_gate_loss_scale = 0.0f;
		net->y_param->flux_refine_final_loss_scale = 0.0f;
		net->y_param->flux_refine_delta_norm_scale = 0.25f;
		net->y_param->flux_refine_gate_margin_norm = 0.01f;
		net->y_param->max_nb_obj_per_image = 0;
	net->y_param->fit_dim = 0;
	
	net->y_param->strict_box_size_association = 0;
	net->y_param->rand_startup = 0;
	net->y_param->rand_prob_best_box_assoc = 0.0f;
	net->y_param->min_prior_forced_scaling = -1.0f;
	
	net->y_param->class_softmax = 0;
	net->y_param->diff_flag = 0;
	net->y_param->no_override = 0;
	net->y_param->raw_output = 0;
	
	
	net->y_param->no_override = no_override;
	net->y_param->raw_output = raw_output;
	if(nb_angle < 0)
		nb_angle = 0;
	angle_target_dim = (nb_angle > 0) ? nb_angle + 1 : 0;
	
	if(max_nb_obj_per_image > 0 && (1+max_nb_obj_per_image*(7+nb_param+angle_target_dim+diff_flag)) != net->output_dim)
	{
		printf("\n ERROR: Network output dim (target) specified in init_network and YOLO's \"max_nb_obj_per_image\" values do not match.\n");
		printf(" Output_dim should be equal to 1+max_nb_obj_per_image*(7+nb_param+nb_angle+angle_weight).\n");
		printf(" Got output_dim = %d, and max_nb_obj_per_image = %d \n\n", net->output_dim, max_nb_obj_per_image);
		exit(EXIT_FAILURE);
	}
	
	if(strcmp(IoU_type_char, "IoU") == 0)
	{
		net->y_param->IoU_type = IOU;
		sprintf(display_IoU_type_char, "Classical IoU");
		net->y_param->c_IoU_fct = IoU_fct;
	}
	else if(strcmp(IoU_type_char, "GIoU") == 0)
	{
		net->y_param->IoU_type = GIOU;
		sprintf(display_IoU_type_char, "Generalized GIoU");
		net->y_param->c_IoU_fct = GIoU_fct;
	}
	else if(strcmp(IoU_type_char, "DIoU") == 0)
	{
		net->y_param->IoU_type = DIOU;
		sprintf(display_IoU_type_char, "Distance DIoU");
		net->y_param->c_IoU_fct = DIoU_fct;
	}
	else if(strcmp(IoU_type_char, "DIoU2") == 0)
	{
		net->y_param->IoU_type = DIOU2;
		sprintf(display_IoU_type_char, "Distance DIoU2");
		net->y_param->c_IoU_fct = DIoU2_fct;
	}
	else if(strcmp(IoU_type_char, "RotIoU") == 0 ||
			strcmp(IoU_type_char, "OBB") == 0 ||
			strcmp(IoU_type_char, "OBB_IoU") == 0)
	{
		net->y_param->IoU_type = ROTIOU;
		sprintf(display_IoU_type_char, "Rotated OBB IoU");
		net->y_param->c_IoU_fct = IoU_fct;
	}
	else
	{
		printf(" WARNING: Unrecognized IoU type: %s, fallback to default GIoU\n", IoU_type_char);
		net->y_param->IoU_type = GIOU;
		sprintf(display_IoU_type_char, "Generalized GIoU");
		net->y_param->c_IoU_fct = GIoU_fct;
	}
	
	if(strcmp(prior_dist_type_char, "IoU") == 0 ||
		strcmp(prior_dist_type_char, "IOU") == 0)
	{
		net->y_param->prior_dist_type = DIST_IOU;
		sprintf(display_prior_dist_type, "Prior dist. IoU");
	}
	else if(strcmp(prior_dist_type_char, "SIZE") == 0)
	{
		net->y_param->prior_dist_type = DIST_SIZE;
		sprintf(display_prior_dist_type, "Prior dist. SIZE");
	}
	else if(strcmp(prior_dist_type_char, "OFFSET") == 0)
	{
		net->y_param->prior_dist_type = DIST_OFFSET;
		sprintf(display_prior_dist_type, "Prior dist. OFFSET");
	}
	else
	{
		printf(" WARNING: Unrecognized prior dist. type: %s, fallback to default dist. Size\n", prior_dist_type_char);
		net->y_param->prior_dist_type = DIST_SIZE;
		sprintf(display_prior_dist_type, "Prior dist. SIZE");
	}
	
	net->y_param->strict_box_size_association = strict_box_size;
	
	if(rand_startup < 0)
		net->y_param->rand_startup = 64000;
	else
		net->y_param->rand_startup = rand_startup;
	
	if(rand_prob_best_box_assoc < 0.0f)
		net->y_param->rand_prob_best_box_assoc = 0.0f;
	else
		net->y_param->rand_prob_best_box_assoc = rand_prob_best_box_assoc;
		
	if(rand_prob < 0.0f)
		net->y_param->rand_prob = 0.0f;
	else
		net->y_param->rand_prob = rand_prob;
	
	if(min_prior_forced_scaling <= 0.0f)
		net->y_param->min_prior_forced_scaling = 0.0f;
	else
		net->y_param->min_prior_forced_scaling = min_prior_forced_scaling;
	
	
	net->y_param->fit_dim = fit_dim;
	if(prior_size == NULL)
	{
		prior_size = (float*) calloc(3*nb_box, sizeof(float));
		for(i = 0; i < 3*nb_box; i++)
			prior_size[i] = 0.0f;
	}
	else
	{
		for(i = 0; i < 3*nb_box; i++)
			prior_size[i] = prior_size[i];
	}
	
	if(yolo_noobj_prob_prior == NULL)
	{
		yolo_noobj_prob_prior = (float*) calloc(nb_box, sizeof(float));
		for(i = 0; i < nb_box; i++)
			yolo_noobj_prob_prior[i] = 0.2f;
	}
	
	l_scale_tab = (float*) calloc(6,sizeof(float));

	l_scale_tab[0] = 2.0f; /*Pos */  l_scale_tab[1] = 2.0f; /*Size */
	l_scale_tab[2] = 1.0f; /*Proba*/ l_scale_tab[3] = 2.0f; /*Objct*/
	l_scale_tab[4] = 1.0f; /*Class*/ l_scale_tab[5] = 1.0f; /*Param*/

	if(scale_tab != NULL)
		for(i = 0; i < 6; i++)
			if(scale_tab[i] > 0.0f)
				l_scale_tab[i] = scale_tab[i];
	
	
	temp = (float*) calloc(6*3, sizeof(float));
	l_slopes_and_maxes_tab = (float**) malloc(6*sizeof(float*));
	for(i = 0; i < 6; i++)
		l_slopes_and_maxes_tab[i] = &temp[i*3];
	
	sm = l_slopes_and_maxes_tab;
	sm[0][0] = 1.0f; sm[0][1] = 6.0f; sm[0][2] = -6.0f;
	sm[1][0] = 1.0f; sm[1][1] = 1.6f; sm[1][2] = -1.6f;
	sm[2][0] = 1.0f; sm[2][1] = 6.0f; sm[2][2] = -6.0f;
	sm[3][0] = 1.0f; sm[3][1] = 6.0f; sm[3][2] = -6.0f;
	sm[4][0] = 1.0f; sm[4][1] = 6.0f; sm[4][2] = -6.0f;
	sm[5][0] = 1.0f; sm[5][1] = 1.2f; sm[5][2] = -0.2f;
	
	if(slopes_and_maxes_tab != NULL)
	{
		for(i = 0; i < 6; i++)
		{
			if(slopes_and_maxes_tab[i][0] > 0.0f)
				sm[i][0] = slopes_and_maxes_tab[i][0];
			if(slopes_and_maxes_tab[i][1] < 100000.0f)
				sm[i][1] = slopes_and_maxes_tab[i][1];
			if(slopes_and_maxes_tab[i][2] > -100000.0f)
				sm[i][2] = slopes_and_maxes_tab[i][2];
		}
	}
	
	if(param_ind_scale == NULL)
	{
		param_ind_scale = (float*) calloc(nb_param, sizeof(float));
		for(i = 0; i < nb_param; i++)
			param_ind_scale[i] = 1.0f;
	}

	l_IoU_limits = (float*) calloc(8,sizeof(float));
	switch(net->y_param->IoU_type)
	{
		case IOU:
		case ROTIOU:
			l_IoU_limits[0] =  0.5f;  l_IoU_limits[1] =  0.1f;
			l_IoU_limits[2] =  0.0f;  l_IoU_limits[3] =  0.0f;
			l_IoU_limits[4] =  0.2f;  l_IoU_limits[5] =  0.2f;
			l_IoU_limits[6] =  0.5f;  l_IoU_limits[7] =  0.3f;
			break;
		
		default:
		case GIOU:
			l_IoU_limits[0] =  0.4f; l_IoU_limits[1] = -0.5f;
			l_IoU_limits[2] = -1.0f; l_IoU_limits[3] = -1.0f;
			l_IoU_limits[4] = -0.3f; l_IoU_limits[5] = -0.3f;
			l_IoU_limits[6] =  0.4f; l_IoU_limits[7] =  0.2f;
			break;
		
		case DIOU:
			l_IoU_limits[0] =  0.3f; l_IoU_limits[1] = -0.6f;
			l_IoU_limits[2] = -1.0f; l_IoU_limits[3] = -1.0f;
			l_IoU_limits[4] = -0.5f; l_IoU_limits[5] = -0.5f;
			l_IoU_limits[6] =  0.3f; l_IoU_limits[7] =  0.1f;
			break;
			
		case DIOU2:
			l_IoU_limits[0] =  0.3f; l_IoU_limits[1] = -0.5f;
			l_IoU_limits[2] = -1.0f; l_IoU_limits[3] = -1.0f;
			l_IoU_limits[4] = -0.4f; l_IoU_limits[5] = -0.4f;
			l_IoU_limits[6] =  0.3f; l_IoU_limits[7] =  0.1f;
			break;
		
	}
	
	if(IoU_limits != NULL)
		for(i = 0; i < 8; i++)
			if(IoU_limits[i] > -1.99f)
				l_IoU_limits[i] = IoU_limits[i];
	
	
	l_fit_parts = (int*) calloc(6,sizeof(int));
	l_fit_parts[0] = 1; /*Position */ 
	l_fit_parts[1] = 1; /*Size */ 
	l_fit_parts[2] = 1; /*Prob */
	l_fit_parts[3] = 1; /*Object*/
	l_fit_parts[4] = 1; /*Class*/
	l_fit_parts[5] = 1; /*Param*/
	
	if(nb_class <= 0)
		l_fit_parts[4] = -1;
	if(nb_param <= 0) /*Param*/
		l_fit_parts[5] = -1;
	
	if(fit_parts != NULL)
		for(i = 0; i < 6; i++)
			if(fit_parts[i] > -2)
				l_fit_parts[i] = fit_parts[i];
	
	net->y_param->nb_box = nb_box;
	net->y_param->prior_size = prior_size;
	net->y_param->noobj_prob_prior = yolo_noobj_prob_prior;
	net->y_param->nb_class = nb_class;
	net->y_param->nb_param = nb_param;
	net->y_param->nb_angle = nb_angle;
	net->y_param->max_nb_obj_per_image = max_nb_obj_per_image;
	net->y_param->class_softmax = class_softmax;
	net->y_param->diff_flag = diff_flag;
		net->y_param->fit_angle = (nb_angle > 0) ? fit_angle : -1;
		net->y_param->angle_loss_mode = (nb_angle > 0 && angle_loss_mode > 0) ? angle_loss_mode : 0;
		net->y_param->target_box_mode = (target_box_mode > 0) ? target_box_mode : 0;
		net->y_param->obb_loss_mode = (nb_angle >= 2 && obb_loss_mode > 0 && obb_loss_scale > 0.0f) ? obb_loss_mode : 0;
		net->y_param->obb_loss_scale = (net->y_param->obb_loss_mode > 0) ? obb_loss_scale : 0.0f;
		net->y_param->multi_pos_topk = (multi_pos_topk > 1) ? multi_pos_topk : 1;
		net->y_param->multi_pos_iou_ratio = (multi_pos_iou_ratio > 0.0f && multi_pos_iou_ratio <= 1.0f) ? multi_pos_iou_ratio : 0.60f;
		net->y_param->multi_pos_min_iou = (multi_pos_min_iou > -1.99f && multi_pos_min_iou < 1.0f) ? multi_pos_min_iou : 0.05f;
		net->y_param->multi_pos_obj_weight = (multi_pos_obj_weight > 0.0f && multi_pos_obj_weight <= 1.0f) ? multi_pos_obj_weight : 0.35f;
		net->y_param->prob_quality_mode = (prob_quality_mode > 0) ? prob_quality_mode : 0;
		net->y_param->obj_quality_mode = (obj_quality_mode > 0) ? obj_quality_mode : 0;
	net->y_param->prob_quality_scale = (prob_quality_scale > 0.0f) ? prob_quality_scale : 1.0f;
	net->y_param->prob_quality_floor = (prob_quality_floor > 0.0f && prob_quality_floor < 0.5f) ? prob_quality_floor : 0.02f;
	net->y_param->obj_quality_scale = (obj_quality_scale > 0.0f) ? obj_quality_scale : 1.0f;
	net->y_param->obj_quality_floor = (obj_quality_floor > 0.0f && obj_quality_floor < 0.5f) ? obj_quality_floor : 0.02f;
	net->y_param->obj_quality_center_weight = (obj_quality_center_weight >= 0.0f) ? obj_quality_center_weight : 0.60f;
	net->y_param->obj_quality_geom_weight = (obj_quality_geom_weight >= 0.0f) ? obj_quality_geom_weight : 0.25f;
	net->y_param->obj_quality_phys_weight = (obj_quality_phys_weight >= 0.0f) ? obj_quality_phys_weight : 0.15f;
	net->y_param->obj_quality_da_center_weight = (obj_quality_da_center_weight >= 0.0f) ? obj_quality_da_center_weight : 0.65f;
	net->y_param->obj_quality_da_geom_weight = (obj_quality_da_geom_weight >= 0.0f) ? obj_quality_da_geom_weight : 0.35f;
	net->y_param->obj_quality_da_phys_weight = (obj_quality_da_phys_weight >= 0.0f) ? obj_quality_da_phys_weight : 0.0f;
	net->y_param->obj_quality_da_flux_max = (obj_quality_da_flux_max >= 0.0f && obj_quality_da_flux_max <= 1.0f) ? obj_quality_da_flux_max : 0.20f;
	net->y_param->obj_quality_da_aspect_min = (obj_quality_da_aspect_min >= 1.0f) ? obj_quality_da_aspect_min : 1.5f;
	net->y_param->obj_quality_da_aspect_max = (obj_quality_da_aspect_max >= net->y_param->obj_quality_da_aspect_min) ? obj_quality_da_aspect_max : 3.0f;
	net->y_param->obj_quality_da_bmaj_log_max = obj_quality_da_bmaj_log_max;
		net->y_param->obj_quality_da_bmaj_log_min = obj_quality_da_bmaj_log_min;
		net->y_param->obj_quality_da_bmin_log_max = obj_quality_da_bmin_log_max;
		net->y_param->obj_quality_da_bmin_log_min = obj_quality_da_bmin_log_min;
		net->y_param->scorer_aux_flux_log_max = scorer_aux_flux_log_max;
		net->y_param->scorer_aux_flux_log_min = scorer_aux_flux_log_min;
		net->y_param->scorer_aux_mode = (scorer_aux_mode > 0) ? scorer_aux_mode : 0;
		net->y_param->scorer_aux_center_scale = (scorer_aux_center_scale > 0.0f) ? scorer_aux_center_scale : 0.0f;
		net->y_param->scorer_aux_flux_scale = (scorer_aux_flux_scale > 0.0f) ? scorer_aux_flux_scale : 0.0f;
		net->y_param->scorer_aux_size_scale = (scorer_aux_size_scale > 0.0f) ? scorer_aux_size_scale : 0.0f;
		net->y_param->scorer_aux_da_weight = (scorer_aux_da_weight > 0.0f) ? scorer_aux_da_weight : 1.0f;
		net->y_param->scorer_aux_pixel_arcsec = (scorer_aux_pixel_arcsec > 0.0f) ? scorer_aux_pixel_arcsec : 0.6042492f;
		net->y_param->scorer_aux_beam_arcsec = (scorer_aux_beam_arcsec > 0.0f) ? scorer_aux_beam_arcsec : 0.625f;
		net->y_param->flux_refine_mode = (flux_refine_mode > 0) ? flux_refine_mode : 0;
		net->y_param->flux_refine_delta_param_index = (flux_refine_delta_param_index >= 0) ? flux_refine_delta_param_index : 3;
		net->y_param->flux_refine_gate_param_index = (flux_refine_gate_param_index >= 0) ? flux_refine_gate_param_index : 4;
		net->y_param->flux_refine_detach_base = (flux_refine_detach_base != 0) ? 1 : 0;
		net->y_param->flux_refine_loss_scale = (flux_refine_loss_scale > 0.0f) ? flux_refine_loss_scale : 0.0f;
		net->y_param->flux_refine_gate_loss_scale = (flux_refine_gate_loss_scale > 0.0f) ? flux_refine_gate_loss_scale : 0.0f;
		net->y_param->flux_refine_final_loss_scale = (flux_refine_final_loss_scale > 0.0f) ? flux_refine_final_loss_scale : 0.0f;
		net->y_param->flux_refine_delta_norm_scale = (flux_refine_delta_norm_scale > 0.0f) ? flux_refine_delta_norm_scale : 0.25f;
		net->y_param->flux_refine_gate_margin_norm = (flux_refine_gate_margin_norm > 0.0f) ? flux_refine_gate_margin_norm : 0.01f;
		if((net->y_param->obj_quality_center_weight + net->y_param->obj_quality_geom_weight + net->y_param->obj_quality_phys_weight) <= 1.0e-6f)
	{
		net->y_param->obj_quality_center_weight = 0.60f;
		net->y_param->obj_quality_geom_weight = 0.25f;
		net->y_param->obj_quality_phys_weight = 0.15f;
	}
	if((net->y_param->obj_quality_da_center_weight + net->y_param->obj_quality_da_geom_weight + net->y_param->obj_quality_da_phys_weight) <= 1.0e-6f)
	{
		net->y_param->obj_quality_da_center_weight = 0.65f;
		net->y_param->obj_quality_da_geom_weight = 0.35f;
		net->y_param->obj_quality_da_phys_weight = 0.0f;
	}
	net->y_param->angle_scale = (angle_scale > 0.0f) ? angle_scale : 1.0f;
	net->y_param->angle_unit_norm_scale = (angle_unit_norm_scale > 0.0f) ? angle_unit_norm_scale : 0.0f;
	net->y_param->angle_sm[0] = (angle_slope > 0.0f) ? angle_slope : 1.0f;
	net->y_param->angle_sm[1] = (angle_fmax > -100000.0f && angle_fmax < 100000.0f) ? angle_fmax : 1.2f;
	net->y_param->angle_sm[2] = (angle_fmin > -100000.0f && angle_fmin < 100000.0f) ? angle_fmin : -1.2f;
	net->y_param->min_angle_IoU_lim = (min_angle_IoU_lim > -1.99f) ? min_angle_IoU_lim : l_IoU_limits[5];
	
	if(net->y_param->class_softmax == 0)
		sprintf(display_class_type,"sigmoid-MSE");
	else
		sprintf(display_class_type,"softmax-CrossEntropy");
	
	if(net->y_param->diff_flag == 0)
		sprintf(display_difficult,"False");
	else
		sprintf(display_difficult,"True");
	
	//Priors table must be sent to GPU memory if C_CUDA
	net->y_param->scale_tab = l_scale_tab;
	net->y_param->slopes_and_maxes_tab = l_slopes_and_maxes_tab;
	net->y_param->param_ind_scale = param_ind_scale;
	net->y_param->IoU_limits = l_IoU_limits;
	net->y_param->fit_parts = l_fit_parts;
	
	if(strcmp(error_type, "complete") == 0)
	{
		net->y_param->error_type = ERR_COMPLETE;
		sprintf(display_error_type, "COMPLETE");
	}
	else if(strcmp(error_type, "natural") == 0)
	{
		net->y_param->error_type = ERR_NATURAL;
		sprintf(display_error_type, "NATURAL");
	}
	else
	{
		printf(" WARNING: Unrecognized YOLO display error type %s, fallback to default \"natural\"\n", error_type);
		net->y_param->error_type = ERR_NATURAL;
		sprintf(display_error_type, "NATURAL");
	}
	
	printf("\n YOLO layer setup \n");
	printf(" -------------------------------------------------------------------\n");
	printf(" Nboxes = %d\n Nclasses = %d\n Nparams = %d\n Nangle = %d\n IoU type = %s\n",
			net->y_param->nb_box, net->y_param->nb_class, net->y_param->nb_param, net->y_param->nb_angle, display_IoU_type_char);
	printf(" Classification type: %s\n", display_class_type);
	printf(" Nb dim fitted : %d\n\n",net->y_param->fit_dim);
	printf(" W priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%4.4f ", net->y_param->prior_size[i*3+0]);
	printf("]\n H priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%4.4f ", net->y_param->prior_size[i*3+1]);
	printf("]\n D priors = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%4.4f ", net->y_param->prior_size[i*3+2]);
	printf("]\n");
	printf(" No obj. prob. priors\n          = [");
	for(i = 0; i < net->y_param->nb_box; i++)
		printf("%4.4f ", net->y_param->noobj_prob_prior[i]);
	printf("]\n");
	printf(" Fit parts: (Pos., Size, Prob., Obj., Class., Param.)\n   = [");
	for(i = 0; i < 6; i++)
		printf(" %d ",net->y_param->fit_parts[i]);
	printf("]\n");
	printf(" Error scales: (Pos., Size, Prob., Obj., Class., Param.)\n   = [");
	for(i = 0; i < 6; i++)
		printf(" %5.3f ",net->y_param->scale_tab[i]);
	printf("]\n");
	printf(" IoU lim.: (GdNotBest, LowBest, Prob., Obj., Class., Param., diffIoUlim, diffObjlim)\n   = [");
	for(i = 0; i < 8; i++)
		printf("%7.3f ", net->y_param->IoU_limits[i]);
	printf("]\n");
	
	if(net->y_param->nb_param > 0)
	{
		printf(" Individual param. error scaling: \n   = [");
		for(i = 0; i < net->y_param->nb_param; i++)
			printf("%7.3f ", net->y_param->param_ind_scale[i]);
		printf("]\n");
	}
		if(net->y_param->nb_angle > 0)
		{
			printf(" Angle head: channels=%d fit=%d mode=%d scale=%7.3f unit_norm=%7.3f minIoU=%7.3f\n",
				net->y_param->nb_angle, net->y_param->fit_angle, net->y_param->angle_loss_mode, net->y_param->angle_scale,
				net->y_param->angle_unit_norm_scale, net->y_param->min_angle_IoU_lim);
			printf(" Angle activation slope/limits: [%6.2f %6.2f %6.2f]\n",
				net->y_param->angle_sm[0], net->y_param->angle_sm[1], net->y_param->angle_sm[2]);
		}
		printf(" Target box mode: %s\n", net->y_param->target_box_mode > 0 ? "CENTER_SIZE" : "INTERVAL_ENDPOINTS");
		if(net->y_param->obb_loss_mode > 0)
		{
			printf(" OBB geometry auxiliary loss: mode=%d scale=%7.4f\n",
				net->y_param->obb_loss_mode, net->y_param->obb_loss_scale);
		}
		if(net->y_param->prob_quality_mode > 0)
		{
			printf(" Probability quality mode: mode=%d scale=%7.3f floor=%7.3f\n",
				net->y_param->prob_quality_mode, net->y_param->prob_quality_scale, net->y_param->prob_quality_floor);
		}
			if(net->y_param->obj_quality_mode > 0)
			{
			printf(" Objectness quality mode: mode=%d scale=%7.3f floor=%7.3f weights=[%5.3f %5.3f %5.3f]\n",
				net->y_param->obj_quality_mode, net->y_param->obj_quality_scale, net->y_param->obj_quality_floor,
				net->y_param->obj_quality_center_weight, net->y_param->obj_quality_geom_weight,
				net->y_param->obj_quality_phys_weight);
			if(net->y_param->obj_quality_mode == 2)
			{
				printf(" Objectness D&A-like weights=[%5.3f %5.3f %5.3f] flux_norm<=%6.4f aspect=[%5.2f,%5.2f]\n",
					net->y_param->obj_quality_da_center_weight, net->y_param->obj_quality_da_geom_weight,
					net->y_param->obj_quality_da_phys_weight, net->y_param->obj_quality_da_flux_max,
					net->y_param->obj_quality_da_aspect_min, net->y_param->obj_quality_da_aspect_max);
				if(net->y_param->obj_quality_da_bmaj_log_max > net->y_param->obj_quality_da_bmaj_log_min &&
					net->y_param->obj_quality_da_bmin_log_max > net->y_param->obj_quality_da_bmin_log_min)
				{
					printf(" Objectness D&A-like physical aspect uses log bmaj=[%7.3f,%7.3f] log bmin=[%7.3f,%7.3f]\n",
						net->y_param->obj_quality_da_bmaj_log_min, net->y_param->obj_quality_da_bmaj_log_max,
						net->y_param->obj_quality_da_bmin_log_min, net->y_param->obj_quality_da_bmin_log_max);
				}
				}
			}
			if(net->y_param->scorer_aux_mode > 0)
			{
				printf(" Scorer-aligned auxiliary loss: mode=%d center=%7.4f flux=%7.4f size=%7.4f da_weight=%7.3f pixel_arcsec=%7.4f beam_arcsec=%7.4f flux_log=[%7.3f,%7.3f]\n",
					net->y_param->scorer_aux_mode, net->y_param->scorer_aux_center_scale,
					net->y_param->scorer_aux_flux_scale, net->y_param->scorer_aux_size_scale,
					net->y_param->scorer_aux_da_weight, net->y_param->scorer_aux_pixel_arcsec,
					net->y_param->scorer_aux_beam_arcsec, net->y_param->scorer_aux_flux_log_min,
					net->y_param->scorer_aux_flux_log_max);
			}
			if(net->y_param->flux_refine_mode > 0)
			{
				printf(" FluxRefine head: mode=%d delta_param=%d gate_param=%d detach_base=%d delta_loss=%7.4f gate_loss=%7.4f final_loss=%7.4f delta_norm_scale=%7.4f gate_margin_norm=%7.4f\n",
					net->y_param->flux_refine_mode, net->y_param->flux_refine_delta_param_index,
					net->y_param->flux_refine_gate_param_index, net->y_param->flux_refine_detach_base,
					net->y_param->flux_refine_loss_scale, net->y_param->flux_refine_gate_loss_scale,
					net->y_param->flux_refine_final_loss_scale, net->y_param->flux_refine_delta_norm_scale,
					net->y_param->flux_refine_gate_margin_norm);
			}
			if(net->y_param->multi_pos_topk > 1)
		{
			printf(" Multi-positive assignment: topk=%d iou_ratio=%7.3f min_iou=%7.3f obj_weight=%7.3f\n",
				net->y_param->multi_pos_topk, net->y_param->multi_pos_iou_ratio,
				net->y_param->multi_pos_min_iou, net->y_param->multi_pos_obj_weight);
		}
		
		printf("\n Activation slopes and limits: \n   = ");
	for(i = 0; i < 6; i++)
		printf("[%6.2f %6.2f %6.2f]\n     ", 
			net->y_param->slopes_and_maxes_tab[i][0],
			net->y_param->slopes_and_maxes_tab[i][1],
			net->y_param->slopes_and_maxes_tab[i][2]);
	
	printf("\n *** Other training hyper-parameters *** \n");
	if(net->y_param->strict_box_size_association > 0)
	{
		printf("  Strict box size association is ENABLED\n");
		printf("  Strict association Nb. good priors = %d\n", 
			net->y_param->strict_box_size_association);
	}
	else
	{	
		printf("  Strict box size association is DISABLED\n");
	}
	printf("  Startup random association Nb. item : %d\n", net->y_param->rand_startup);
	printf("  Proportion of forced best prior assoc.: %5.3f\n", net->y_param->rand_prob_best_box_assoc);
	printf("  Proportion of forced random prior assoc.: %5.3f\n", net->y_param->rand_prob);
	printf("  Forced smallest prior association scaling : %6.3f\n", net->y_param->min_prior_forced_scaling);
	printf("  Closest prior association type : %s\n", display_prior_dist_type);
	printf("  Difficult flag in use: %s\n", display_difficult);
	printf("  Display error type : %s\n", display_error_type);
	printf("\n -------------------------------------------------------------------\n\n");
	
	return (net->y_param->nb_box * (8 + net->y_param->nb_class + net->y_param->nb_param + net->y_param->nb_angle));
}

void free_yolo_params(network *net)
{
	yolo_param *y_param = (yolo_param*) net->y_param;
	
	free(y_param->prior_size);
	free(y_param->noobj_prob_prior);
	free(y_param->scale_tab);
	free(y_param->slopes_and_maxes_tab[0]);
	free(y_param->slopes_and_maxes_tab);
	free(y_param->param_ind_scale);
	free(y_param->IoU_limits);
	free(y_param->fit_parts);
	
	free(y_param);
	net->y_param = NULL;
}


void YOLO_activation_fct(void *i_tab, int flat_offset, int len, yolo_param y_param, size_t size, int class_softmax)
{	
	float *tab = (float*) i_tab;
	
	int nb_class = y_param.nb_class, nb_param = y_param.nb_param, nb_angle = y_param.nb_angle;
	/*Default values are in activ_function.c (set_yolo_config)*/
	float **sm_tab = y_param.slopes_and_maxes_tab;
	int fit_dim = y_param.fit_dim;	
	size_t i, col, in_col;
	int output_offset = 8+nb_class+nb_param+nb_angle;
	
	#pragma omp parallel for private(col, in_col) schedule(guided,4)
	for(i = 0; i < size; i++)
	{
		float normal = 0.0f, vmax;
		int j;
		col = i / flat_offset;
		in_col = col%output_offset;
		
		/*Position*/
		if(in_col >= 0 && in_col < 3)	
		{
			if(fit_dim > in_col)
			{	
				tab[i] = -sm_tab[0][0]*tab[i];
				if(tab[i] > sm_tab[0][1])
					tab[i] = sm_tab[0][1];	
				else if(tab[i] < sm_tab[0][2])
					tab[i] = sm_tab[0][2];	
				tab[i] = 1.0f/(1.0f + expf(tab[i]));
			}	
			else
				tab[i] = 0.5f; /*Center of the cell*/
			continue;
		}
	
		/*Box size*/
		if(in_col >= 3 && in_col < 6)	
		{
			if(fit_dim > in_col-3)
			{	
				tab[i] = sm_tab[1][0]*tab[i];	
				if(tab[i] > sm_tab[1][1])
					tab[i] = sm_tab[1][1];	
				else if(tab[i] < (sm_tab[1][2]))
					tab[i] = (sm_tab[1][2]);
			}	
			else
				tab[i] = 0.0f; /*Output = prior*/	
			continue;
		}
	
		/*Object probability*/
		if(in_col == 6)
		{
			tab[i] = -sm_tab[2][0]*tab[i];	
			if(tab[i] > sm_tab[2][1])
				tab[i] = sm_tab[2][1];
			else if(tab[i] < sm_tab[2][2])	
				tab[i] = sm_tab[2][2];
			tab[i] = 1.0f/(1.0f + expf(tab[i]));	
			continue;
		}
	
		/*Objectness (Obj. quality => based on IoU)*/
		if(in_col == 7)
		{
			tab[i] = -sm_tab[3][0]*tab[i];	
			if(tab[i] > sm_tab[3][1])
				tab[i] = sm_tab[3][1];
			else if(tab[i] < sm_tab[3][2])	
				tab[i] = sm_tab[3][2];
			tab[i] = 1.0f/(1.0f + expf(tab[i]));	
			continue;
		}
	
		/*Classes*/
		if(in_col >= 8 && in_col < 8+nb_class)
		{
			if(class_softmax)
			{
				if(in_col != 8)
					continue;
				vmax = tab[i];
				for(j = 1; j < nb_class; j++)
					if(tab[i+j*flat_offset] > vmax)
						vmax = tab[i+j*flat_offset];
				
				for(j = 0; j < nb_class; j++)
				{
					tab[i+j*flat_offset] = expf((tab[i+j*flat_offset]-vmax));
					normal += (float)tab[i+j*flat_offset];
				}
				
				for(j = 0; j < nb_class; j++)
					tab[i+j*flat_offset] = ((float)tab[i+j*flat_offset]/normal);
			}
			else
			{
				tab[i] = -sm_tab[4][0]*tab[i];	
				if(tab[i] > sm_tab[4][1])
					tab[i] = sm_tab[4][1];
				else if(tab[i] < sm_tab[4][2])	
					tab[i] = sm_tab[4][2];
				tab[i] = 1.0f/(1.0f + expf(tab[i]));
			}
			continue;
		}
	
		/*Additional parameters (regression)*/
		if(in_col >= 8+nb_class && in_col < 8+nb_class+nb_param)
		{
			tab[i] = sm_tab[5][0]*tab[i];
			if(tab[i] > sm_tab[5][1])
				tab[i] = sm_tab[5][1];
			else if(tab[i] < (sm_tab[5][2]))	
				tab[i] = (sm_tab[5][2]);
			continue;
		}	
		/*Angle head regression*/
		if(in_col >= 8+nb_class+nb_param)
		{
			tab[i] = y_param.angle_sm[0]*tab[i];
			if(tab[i] > y_param.angle_sm[1])
				tab[i] = y_param.angle_sm[1];
			else if(tab[i] < y_param.angle_sm[2])
				tab[i] = y_param.angle_sm[2];
			continue;
		}
	}
}


void YOLO_deriv_error_fct
	(void *i_delta_o, void *i_output, void *i_target, int flat_target_size, int flat_output_size,
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size, int nb_im_iter)
{
	float *t_delta_o = (float*) i_delta_o;
	float *t_output = (float*) i_output;
	float *t_target = (float*) i_target;

	/* Define many "shorts" for y_param content to enhance code redeability*/
	int nb_box                      = y_param.nb_box; 
	int nb_class                    = y_param.nb_class;
	int nb_param                    = y_param.nb_param; 
	int nb_angle                    = y_param.nb_angle;
	int strict_box_size_association = y_param.strict_box_size_association;
	int fit_dim                     = y_param.fit_dim;
	int rand_startup                = y_param.rand_startup;
	float rand_prob_best_box_assoc  = y_param.rand_prob_best_box_assoc;
	float rand_prob                 = y_param.rand_prob;
	float min_prior_forced_scaling  = y_param.min_prior_forced_scaling;
	int class_softmax               = y_param.class_softmax;
	int diff_flag                   = y_param.diff_flag;
	int prior_dist_type             = y_param.prior_dist_type;
		int target_box_mode             = y_param.target_box_mode;
		int obb_loss_mode               = y_param.obb_loss_mode;
		int multi_pos_topk              = y_param.multi_pos_topk;
		
		float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];
		float angle_scale = y_param.angle_scale, angle_unit_norm_scale = y_param.angle_unit_norm_scale;
		float obb_loss_scale = y_param.obb_loss_scale;
		float multi_pos_iou_ratio = y_param.multi_pos_iou_ratio;
		float multi_pos_min_iou = y_param.multi_pos_min_iou;
		float multi_pos_obj_weight = y_param.multi_pos_obj_weight;

	float *prior_size         = y_param.prior_size;
	int   *cell_size          = y_param.cell_size;
	float *param_ind_scale    = y_param.param_ind_scale;
	float *lambda_noobj_prior = y_param.noobj_prob_prior;
	float **sm_tab            = y_param.slopes_and_maxes_tab;
	int   *t_target_cell_mask = y_param.target_cell_mask;
	float *t_IoU_table        = y_param.IoU_table;
	float *t_dist_prior       = y_param.dist_prior;
	int   *t_box_locked       = y_param.box_locked;
	float *t_box_in_pix       = y_param.box_in_pix;
	
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim      = y_param.IoU_limits[0], low_IoU_best_box_assoc = y_param.IoU_limits[1];
	float min_prob_IoU_lim  = y_param.IoU_limits[2], min_obj_IoU_lim        = y_param.IoU_limits[3];
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim      = y_param.IoU_limits[5];
	float min_angle_IoU_lim = y_param.min_angle_IoU_lim;
	float diff_IoU_lim      = y_param.IoU_limits[6], diff_obj_lim           = y_param.IoU_limits[7];
	int fit_pos = y_param.fit_parts[0], fit_size  = y_param.fit_parts[1], fit_prob  = y_param.fit_parts[2];
	int fit_obj = y_param.fit_parts[3], fit_class = y_param.fit_parts[4], fit_param = y_param.fit_parts[5];
	int fit_angle = y_param.fit_angle;
	int angle_loss_mode = y_param.angle_loss_mode;
	
	#pragma omp parallel
	#ifdef OPEN_MP
	{
	srand((int)time(NULL) ^ omp_get_thread_num());
	#endif
	#pragma omp for schedule(guided,4)
	for(int c_pix = 0; c_pix < size; c_pix++)
	{
		//All private variables inside the loop for convenience
		//Should be marginal since one iteration cost is already high
		
		float *delta_o, *output, *target;
		int *target_cell_mask, *box_locked;
		float *IoU_table, *dist_prior, *box_in_pix;
		int i, j, k, l, l_o, l_t;
		size_t f_offset, c_total_nb_area, c_total_nb_area_batch, total_cell_pos_nb_area, total_area_and_cell_offset;
		int c_batch, output_offset, target_offset, nb_obj_target, s_p_i = 0;
			int nb_in_cell, id_in_cell, id_in_cell_offset, l_r_b = -1, resp_box = -1, resp_targ = -1, resp_targ_offset, targ_diff_flag = 0;
			int secondary_iter, secondary_box;
			float best_dist, c_dist, max_IoU, current_IoU, prob_target, obj_target, multi_pos_iou_thr, secondary_iou, best_secondary_iou;
		int cell_pos[3], c_nb_area[3], obj_c[3];
		float *c_box_in_pix, *c_prior_size;
			float obj_in_offset[6], out_int[6], targ_int[6], targ_size[3];
			float obb_grad_logw, obb_grad_logh, obb_grad_theta, obb_theta_norm2;
				float obb_pred_theta, obb_targ_theta, obb_targ_cx, obb_targ_cy;
			float angle_weight, angle_norm, angle_norm_diff, angle_norm_grad;
			float angle_out0, angle_out1, angle_targ0, angle_targ1, angle_dot, angle_common, angle_norm2;
		float class_only_IoU = -2.0f;
		
		c_nb_area[0] = nb_area_w; c_nb_area[1] = nb_area_h; c_nb_area[2] = nb_area_d;
		c_total_nb_area = c_nb_area[0]*c_nb_area[1]*c_nb_area[2];
		c_batch = c_pix / flat_output_size;
		target = t_target + flat_target_size * c_batch;
		f_offset = size;
		output_offset = 8+nb_class+nb_param+nb_angle;
		target_offset = 7+nb_param+((nb_angle > 0) ? nb_angle + 1 : 0)+diff_flag;
		
		i = c_pix % flat_output_size;
		cell_pos[2] = i / (c_nb_area[0]*c_nb_area[1]);
		cell_pos[1] = (int)(i % (c_nb_area[0]*c_nb_area[1])) / c_nb_area[0];
		cell_pos[0] = (int)(i % (c_nb_area[0]*c_nb_area[1])) % c_nb_area[0];
		
		c_total_nb_area_batch = c_total_nb_area * c_batch;
		total_cell_pos_nb_area = cell_pos[2]*c_nb_area[0]*c_nb_area[1] + cell_pos[1]*c_nb_area[0] + cell_pos[0];
		total_area_and_cell_offset = c_total_nb_area_batch + total_cell_pos_nb_area;
		
		delta_o = t_delta_o + total_area_and_cell_offset;
		output  = t_output  + total_area_and_cell_offset;
		
		target_cell_mask = t_target_cell_mask + total_area_and_cell_offset * y_param.max_nb_obj_per_image;
		IoU_table  = t_IoU_table  + total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;
		dist_prior = t_dist_prior + total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;
		box_locked = t_box_locked + total_area_and_cell_offset * nb_box;
		box_in_pix = t_box_in_pix + total_area_and_cell_offset * 6 * nb_box;
		
		nb_obj_target = target[0];
		target++;
		
		if(nb_obj_target == -1)
		{
			nb_obj_target = 1;
			class_only_IoU = good_IoU_lim;
		}
		
		best_dist = 1000000000;
		for(k = 0; k < nb_box; k++)
		{
			box_locked[k] = 0;
			c_box_in_pix = box_in_pix + k*6;
			c_prior_size = prior_size + k*3;
			l_o = k*output_offset;
			for(l = 0; l < 3; l++)
				c_box_in_pix[l] = ((float)output[(l_o+l)*f_offset] + cell_pos[l]) * cell_size[l];
			for(l = 0; l < 3; l++)
				c_box_in_pix[l+3] = c_prior_size[l]*expf((float)output[(l_o+l+3)*f_offset]);
		
			c_dist = sqrt(c_prior_size[0]*c_prior_size[0] 
						+ c_prior_size[1]*c_prior_size[1] 
						+ c_prior_size[2]*c_prior_size[2]);
			if(c_dist < best_dist)
			{
				best_dist = c_dist;
				s_p_i = k;
			}
		}
		
		nb_in_cell = 0;
		for(j = 0; j < nb_obj_target; j++)
		{
			l_t = j*target_offset;
			yolo_load_target_box(target, l_t, targ_int, target_box_mode);
			
			/* Search for targets that should be predicted by the current cell element */
			target_cell_mask[j] = 1;
			for(l = 0; l < 3; l++)
			{
				obj_c[l] = (int)( (targ_int[l+3] + targ_int[l])*0.5f / cell_size[l]);
				/* If target outside the current cell element, set target flag to 0*/
				if(obj_c[l] != cell_pos[l])
					target_cell_mask[j] = 0;
			}
			
			if(target_cell_mask[j] == 1)
				nb_in_cell++;
			
			/* Flag all the "Good but not best boxes" for all targets regardless of the grid element */
			for(k = 0; k < nb_box; k++)
			{
				l_o = k*output_offset;
				if(box_locked[k] != 0)
					continue;
				c_box_in_pix = box_in_pix+k*6;
				for(l = 0; l < 6; l++)
					out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];
				
				current_IoU = yolo_box_quality(&y_param, out_int, targ_int, output, target,
					l_o, l_t, f_offset, nb_class, nb_param, nb_angle);
				if(current_IoU > good_IoU_lim)
					box_locked[k] = 1;
			}
		}
		
		/* For all target in cell compute the IoU with the prediciton and distance to the prior */
		id_in_cell = 0;
		for(j = 0; j < nb_obj_target; j++)
		{
			id_in_cell_offset = id_in_cell*nb_box;
				if(target_cell_mask[j] == 0)
				{
					continue;
				}
		
				l_t = j*target_offset;
					yolo_load_target_box(target, l_t, targ_int, target_box_mode);
				for(l = 0; l < 3; l++)
				{
					targ_size[l] = targ_int[l+3] - targ_int[l];
				}
				
				for(k = 0; k < nb_box; k++)
				{
					l_o = k*output_offset;
					c_box_in_pix = box_in_pix+k*6;
					for(l = 0; l < 6; l++)
						out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];
					
					current_IoU = yolo_box_quality(&y_param, out_int, targ_int, output, target,
						l_o, l_t, f_offset, nb_class, nb_param, nb_angle);
					IoU_table[id_in_cell_offset + k] = current_IoU;
					dist_prior[id_in_cell_offset + k] = -2.0f;
				}
			
			/* Restrict the association to the l best theoritical prior (times repetition of identical priors) */
			if(strict_box_size_association > 0)
			{
				if(prior_dist_type == DIST_IOU)
					for(l = 0; l < 6; l++)
						targ_int[l] = copysignf(0.5f,l-2.5f)*targ_size[l%3];
			
				for(k = 0; k < nb_box; k++)
				{
					c_prior_size = prior_size + k*3;
					switch(prior_dist_type)
					{
						case DIST_IOU:
							for(l = 0; l < 6; l++)
								out_int[l] = copysignf(0.5f,l-2.5f)*c_prior_size[l%3];
							dist_prior[id_in_cell_offset + k] = 1.0f - y_param.c_IoU_fct(out_int, targ_int);
							break;
						
						default:
						case DIST_SIZE:
							dist_prior[id_in_cell_offset + k] = sqrt(
								 (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])
								+(targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])
								+(targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]));
							break;
						
						case DIST_OFFSET:
							for(l = 0; l < 3; l++)
							{
								obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];
								if(obj_in_offset[l+3] < size_min_sat)
									obj_in_offset[l+3] = logf(size_min_sat);
								else if(obj_in_offset[l+3] > size_max_sat)
									obj_in_offset[l+3] = logf(size_max_sat);
								else
									obj_in_offset[l+3] = logf(obj_in_offset[l+3]);
							}
							
							dist_prior[id_in_cell_offset + k] = 
								 fabs(obj_in_offset[3])
								+fabs(obj_in_offset[4])
								+fabs(obj_in_offset[5]);
							break;
					}
				}
				
				for(l = 0; l < strict_box_size_association; l++)
				{
					best_dist = 1000000.0f;
					for(k = 0; k < nb_box; k++)
						if(dist_prior[id_in_cell_offset+k] > 0.0 && dist_prior[id_in_cell_offset+k] < best_dist)
							best_dist = dist_prior[id_in_cell_offset+k];
					for(k = 0; k < nb_box; k++) /* Flag the closest theoritical prior (and identical ones if any) */
						if(fabs(dist_prior[id_in_cell_offset+k] - best_dist) < 0.001f )
							dist_prior[id_in_cell_offset+k] = -2.0f;
				}
			}
		
			id_in_cell++;
		}
		
		for(id_in_cell = 0; id_in_cell < nb_in_cell; id_in_cell++)
		{
			/* Force a random box association with only criteria being that the box is not already used */
			/* Used as a startup phase to get all the priors closer to the objects to detect */
			if(nb_im_iter <= rand_startup)
			{
				resp_targ = id_in_cell;	resp_box = -1;
				for(k = 0; k < 2*nb_box; k++)
				{
					resp_box = (int)(random_uniform()*nb_box);
					if(box_locked[resp_box] != 2)
						break;
					resp_box = -1;
				}
				
				if(resp_box == -1)
					continue;
				
				l = 0;
				for(j = 0; j < nb_obj_target; j++)
				{
					l += target_cell_mask[j];
					if(l == resp_targ + 1)
						break;
					}
					l_t = j*target_offset;
					resp_targ_offset = resp_targ*nb_box;
				}
			else
			{
				max_IoU = -2.0f; resp_box = -1;	resp_targ = -1;
				for(j = 0; j < nb_in_cell; j++)
					for(k = 0; k < nb_box; k++)
						if(IoU_table[j*nb_box+k] > max_IoU && dist_prior[j*nb_box+k] < -1.0f)
						{
							max_IoU = IoU_table[j*nb_box+k];
							resp_targ = j;																											
							resp_box = k;
						}
				
				/* If strict_box_size > 0 and no more good prior is available, or if there are more targets than boxes */
				/* In that case all the remaining targets are unable to be associated to */
				/* any other box and the id_in_cell loop must be stoped */
				if(resp_box == -1)
					continue;
				
				/* l is the "best" index in the "in cell" list */
				/* Need to get back the original target index from the "in cell" index */
				l = 0;
				for(j = 0; j < nb_obj_target; j++)
				{
					l += target_cell_mask[j];
					if(l == resp_targ + 1)
						break;
				}
				/* The appropriate j value is set after this early stop loop */
				l_t = j*target_offset;
				resp_targ_offset = resp_targ*nb_box;
				
			yolo_load_target_box(target, l_t, targ_int, target_box_mode);
				for(l = 0; l < 3; l++)
					targ_size[l] = targ_int[l+3] - targ_int[l];
					
				if(random_uniform() < rand_prob)
				{
					for(k = 0; k < 2*nb_box; k++)
					{
						l_r_b = (int)(random_uniform()*nb_box);
						if(box_locked[l_r_b] != 2)
						{
							resp_box = l_r_b;
							break;
						}
					}
				}
				/* Force the association to the smallest prior (or identical) if the target is too small */
				else if(targ_size[0] < min_prior_forced_scaling*prior_size[s_p_i*3+0]
					&& targ_size[1] < min_prior_forced_scaling*prior_size[s_p_i*3+1]
					&& targ_size[2] < min_prior_forced_scaling*prior_size[s_p_i*3+2])
				{
					max_IoU = -2.0f; best_dist = prior_size[s_p_i*3+0]*prior_size[s_p_i*3+1]*prior_size[s_p_i*3+2];
					for(k = 0; k < nb_box; k++)
					{
						c_prior_size = prior_size + k*3;
						if((prior_size[s_p_i*3+0] == c_prior_size[0]
							&& prior_size[s_p_i*3+1] == c_prior_size[1]
							&& prior_size[s_p_i*3+2] == c_prior_size[2])
							&& IoU_table[resp_targ_offset + k] > max_IoU)
						{
							max_IoU = IoU_table[resp_targ_offset + k];
							resp_box = k;
						}
					}
				}
				/* If prediction is too bad, associate it it the best theoritical prior instead (might found the same box again) */
				/* Also force the best theoritical prior association at a small rate */
				else if(max_IoU < low_IoU_best_box_assoc ||
					random_uniform() < rand_prob_best_box_assoc)
				{
					if(prior_dist_type == DIST_IOU)
						for(l = 0; l < 6; l++)
							targ_int[l] = copysignf(0.5f,l-2.5f)*targ_size[l%3];
					
					best_dist = 100000.0f;
					for(k = 0; k < nb_box; k++)
					{
						c_prior_size = prior_size + k*3;
						switch(prior_dist_type)
						{
							case DIST_IOU:
								for(l = 0; l < 6; l++)
									out_int[l] = copysignf(0.5f,l-2.5f)*c_prior_size[l%3];
								dist_prior[resp_targ_offset + k] = 1.0f - y_param.c_IoU_fct(out_int, targ_int);
								break;
							
							default:
							case DIST_SIZE:
								dist_prior[resp_targ_offset + k] = sqrt(
									 (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])
									+(targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])
									+(targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]));
								break;
							
							case DIST_OFFSET:
								for(l = 0; l < 3; l++)
								{
									obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];
									if(obj_in_offset[l+3] < size_min_sat)
										obj_in_offset[l+3] = logf(size_min_sat);
									else if(obj_in_offset[l+3] > size_max_sat)
										obj_in_offset[l+3] = logf(size_max_sat);
									else
										obj_in_offset[l+3] = logf(obj_in_offset[l+3]);
								}
								
								dist_prior[resp_targ_offset + k] = 
									 fabs(obj_in_offset[3])
									+fabs(obj_in_offset[4])
									+fabs(obj_in_offset[5]);
								break;
						}
						if(dist_prior[resp_targ_offset + k] < best_dist)
							best_dist = dist_prior[resp_targ_offset + k];
					}
					max_IoU = -2.0f;
					for(k = 0; k < nb_box; k++)
					{
						if(fabsf(dist_prior[resp_targ_offset + k] - best_dist) < 0.001f && IoU_table[resp_targ_offset + k] > max_IoU)
						{
							max_IoU = IoU_table[resp_targ_offset + k];
							resp_box = k;
						}
					}
					/* If the best prior (or identical) is not available, the resp_box is unchanged */
					/* Should always get a resp_box != -1, regarding all previous conditions */
				}
			}
		
					c_box_in_pix = box_in_pix + resp_box*6;
					for(l = 0; l < 6; l++)
						out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];
				
			yolo_load_target_box(target, l_t, targ_int, target_box_mode);
				for(l = 0; l < 3; l++)
					targ_size[l] = targ_int[l+3] - targ_int[l];
				
				l_o = resp_box*output_offset;
				max_IoU = yolo_box_quality(&y_param, out_int, targ_int, output, target,
					l_o, l_t, f_offset, nb_class, nb_param, nb_angle);
				if(max_IoU > 0.98f)
					max_IoU = 0.98f;
				if(class_only_IoU > -2.0f)
					max_IoU = class_only_IoU; /*regardless of actual IoU because class only box is not precise*/
				
				c_prior_size = prior_size + 3*resp_box;
			
			/* Positive reinforcement */
			targ_diff_flag = 0;
			if(diff_flag)	/* Cast from mixed precision type to float is always possible, but not necessary to int directly */
				targ_diff_flag = (int)((float)target[l_t+7+nb_param+((nb_angle > 0) ? nb_angle + 1 : 0)]);
			
			/* If the target is flagged as "difficult", only update the matching box if the prediction is already confident enough */
			/* The target is removed from the list anyway, and the corresponding box fall to "background" or "Good_but_not_best" case*/	
			if(diff_flag && targ_diff_flag > 0 && (max_IoU < diff_IoU_lim || (float)output[(l_o+7)*f_offset] < diff_obj_lim))
			{
				for(k = 0; k < nb_box; k++)
					IoU_table[resp_targ_offset + k] = -2.0f;
				continue;
			}
		
			/* Mark the box as already associated by removing its contributions to the IoU table */
			for(j = 0; j < nb_in_cell; j++)
				IoU_table[j*nb_box + resp_box] = -2.0f;
			
			box_locked[resp_box] = 2;
			
			for(l = 0; l < 3; l++)
				obj_in_offset[l] = ((targ_int[l+3] + targ_int[l])*0.5f - cell_pos[l]*cell_size[l])/(float)cell_size[l];
			for(l = 0; l < 3; l++)
			{
				obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];
				if(obj_in_offset[l+3] < size_min_sat)
					obj_in_offset[l+3] = logf(size_min_sat);
				else if(obj_in_offset[l+3] > size_max_sat)
					obj_in_offset[l+3] = logf(size_max_sat);
				else
					obj_in_offset[l+3] = logf(obj_in_offset[l+3]);
			}
			
			/* Note: most of the following could be replaced by function pointers to avoid so much switch statements */
			switch(fit_pos)
			{
				case 1:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k && class_only_IoU < -1.9f && (diff_flag == 0 || targ_diff_flag < 3))
							delta_o[(l_o+k)*f_offset] = ( sm_tab[0][0]
								*coord_scale*(float)output[(l_o+k)*f_offset]
								*(1.0f-(float)output[(l_o+k)*f_offset])
								*((float)output[(l_o+k)*f_offset]-obj_in_offset[k]));
						else
							delta_o[(l_o+k)*f_offset] = (0.0f);
					}
					break;
				case 0:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k)
							delta_o[(l_o+k)*f_offset] = ( sm_tab[0][0]
								*coord_scale*(float)output[(l_o+k)*f_offset]
								*(1.0f-(float)output[(l_o+k)*f_offset])
								*((float)output[(l_o+k)*f_offset]-0.5f));
						else
							delta_o[(l_o+k)*f_offset] = (0.0f);
					}
					break;
				case -1:
					for(k = 0; k < 3; k++)
						delta_o[(l_o+k)*f_offset] = (0.0f);
					break;		
			}
			
			switch(fit_size)
			{
				case 1:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k && class_only_IoU < -1.9f && (diff_flag == 0 || targ_diff_flag < 3))
							delta_o[(l_o+k+3)*f_offset] = ( sm_tab[1][0]
								*size_scale*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3]));
						else
							delta_o[(l_o+k+3)*f_offset] = (0.0f);
					}
					break;
				case 0:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k)
							delta_o[(l_o+k+3)*f_offset] = ( sm_tab[1][0]
								*size_scale*((float)output[(l_o+k+3)*f_offset]-0.0f));
						else
							delta_o[(l_o+k+3)*f_offset] = (0.0f);
					}
					break;
				case -1:
					for(k = 0; k < 3; k++)
						delta_o[(l_o+k+3)*f_offset] = (0.0f);
					break;
			}
		
			switch(fit_prob)
			{
				case 1:
					if(max_IoU > min_prob_IoU_lim)
					{
							prob_target = yolo_probability_quality_target(&y_param, output, target,
								l_o, l_t, f_offset, nb_class, nb_param, nb_angle, max_IoU, obj_in_offset);
						delta_o[(l_o+6)*f_offset] = ( sm_tab[2][0]
							*prob_scale*(float)output[(l_o+6)*f_offset]
							*(1.0f-(float)output[(l_o+6)*f_offset])
							*((float)output[(l_o+6)*f_offset]-prob_target));
					}
					else
						delta_o[(l_o+6)*f_offset] = (0.0f);
					break;
				case 0:
					delta_o[(l_o+6)*f_offset] = ( sm_tab[2][0]
						*prob_scale*(float)output[(l_o+6)*f_offset]
						*(1.0f-(float)output[(l_o+6)*f_offset])
						*((float)output[(l_o+6)*f_offset]-0.5f));
					break;
				case -1:
					delta_o[(l_o+6)*f_offset] = (0.0f);
					break;
			}
		
			switch(fit_obj)
			{
				case 1:
					if(max_IoU > min_obj_IoU_lim)
					{
						obj_target = yolo_objectness_quality_target(&y_param, output, target,
							l_o, l_t, f_offset, nb_class, nb_param, max_IoU, obj_in_offset);
						delta_o[(l_o+7)*f_offset] = ( sm_tab[3][0]
							*obj_scale*(float)output[(l_o+7)*f_offset]
							*(1.0f-(float)output[(l_o+7)*f_offset])
							*((float)output[(l_o+7)*f_offset]-obj_target));
					}
					else
						delta_o[(l_o+7)*f_offset] = (0.0f);
					break;
				case 0:
					delta_o[(l_o+7)*f_offset] = ( sm_tab[3][0]
						*obj_scale*(float)output[(l_o+7)*f_offset]
						*(1.0f-(float)output[(l_o+7)*f_offset])
						*((float)output[(l_o+7)*f_offset]-0.5f));
					break;
				case -1:
					delta_o[(l_o+7)*f_offset] = (0.0f);
					break;
			}
			
			switch(fit_class)
			{
				case 1:
					if(max_IoU > min_class_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))
					{
						if(class_softmax)
						{
							for(k = 0; k < nb_class; k++)
							{
								if(k == (int) target[l_t]-1)
									delta_o[(l_o+8+k)*f_offset] = (class_scale*((float)output[(l_o+8+k)*f_offset]-1.0f));
								else
									delta_o[(l_o+8+k)*f_offset] = (class_scale*((float)output[(l_o+8+k)*f_offset]-0.0f));
							}
						}
						else
						{
							for(k = 0; k < nb_class; k++)
							{
								if(k == (int) target[l_t]-1)
									delta_o[(l_o+8+k)*f_offset] = ( sm_tab[4][0]
										*class_scale*(float)output[(l_o+8+k)*f_offset]
										*(1.0f-(float)output[(l_o+8+k)*f_offset])
										*((float)output[(l_o+8+k)*f_offset]-0.98f));
								else
									delta_o[(l_o+8+k)*f_offset] = ( sm_tab[4][0]
										*class_scale*(float)output[(l_o+8+k)*f_offset]
										*(1.0f-(float)output[(l_o+8+k)*f_offset])
										*((float)output[(l_o+8+k)*f_offset]-0.02f));
							}
						}
					}
					else
						for(k = 0; k < nb_class; k++)
							delta_o[(l_o+8+k)*f_offset] = (0.0f);
					break;
				case 0:
					if(class_softmax)
					{
						/* Could compute CE with target = 1/nb_class, but in this case perfect classification error > 0 (still minimum) */
						for(k = 0; k < nb_class; k++)
							delta_o[(l_o+8+k)*f_offset] = (0.0f);
					}
					else
					{
						for(k = 0; k < nb_class; k++)
							delta_o[(l_o+8+k)*f_offset] = ( sm_tab[4][0]
								*class_scale*(float)output[(l_o+8+k)*f_offset]
								*(1.0f-(float)output[(l_o+8+k)*f_offset])
								*((float)output[(l_o+8+k)*f_offset]-0.5f));
					}
					break;
				case -1:
					for(k = 0; k < nb_class; k++)
						delta_o[(l_o+8+k)*f_offset] = (0.0f);
					break;
			}
		
			/* Linear activation of additional parameters */
			switch(fit_param)
			{
				case 1:
					if(max_IoU > min_param_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))
						for(k = 0; k < nb_param; k++)
						{
							if(yolo_flux_refine_is_aux_param(&y_param, k, nb_param))
							{
								delta_o[(l_o+8+nb_class+k)*f_offset] = 0.0f;
								continue;
							}
							delta_o[(l_o+8+nb_class+k)*f_offset] = 
								 (param_ind_scale[k]* sm_tab[5][0]*param_scale
								*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k]));
						}
					else
						for(k = 0; k < nb_param; k++)
							delta_o[(l_o+8+nb_class+k)*f_offset] = (0.0f);
					break;
				case 0:
					for(k = 0; k < nb_param; k++)
					{
						if(yolo_flux_refine_is_aux_param(&y_param, k, nb_param))
						{
							delta_o[(l_o+8+nb_class+k)*f_offset] = 0.0f;
							continue;
						}
						delta_o[(l_o+8+nb_class+k)*f_offset] = 
							 (param_ind_scale[k]* sm_tab[5][0]*param_scale
							*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f));
					}
					break;
				case -1:
					for(k = 0; k < nb_param; k++)
						delta_o[(l_o+8+nb_class+k)*f_offset] = (0.0f);
					break;
			}

				yolo_add_scorer_aux_delta(&y_param, delta_o, output, target, l_o, l_t, f_offset,
					nb_class, nb_param, obj_in_offset, cell_size, sm_tab, coord_scale, param_scale,
					param_ind_scale, max_IoU, min_param_IoU_lim, diff_flag, targ_diff_flag);
				yolo_add_flux_refine_delta(&y_param, delta_o, output, target, l_o, l_t, f_offset,
					nb_class, nb_param, sm_tab, param_scale, param_ind_scale,
					max_IoU, min_param_IoU_lim, diff_flag, targ_diff_flag);

				/* Encoded angle head: target layout is [cos2theta, sin2theta, angle_weight]. */
			switch(fit_angle)
			{
				case 1:
					if(nb_angle > 0 && max_IoU > min_angle_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))
					{
							angle_weight = (float)target[l_t+7+nb_param+nb_angle];
							if(angle_loss_mode == 1)
							{
								for(k = 0; k < nb_angle; k += 2)
								{
									if(k + 1 >= nb_angle)
									{
										delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] =
											(angle_weight*y_param.angle_sm[0]*angle_scale
											*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k]));
										continue;
									}
									angle_out0 = (float)output[(l_o+8+nb_class+nb_param+k)*f_offset];
									angle_out1 = (float)output[(l_o+8+nb_class+nb_param+k+1)*f_offset];
									angle_targ0 = (float)target[l_t+7+nb_param+k];
									angle_targ1 = (float)target[l_t+7+nb_param+k+1];
									angle_norm = sqrtf(angle_out0*angle_out0 + angle_out1*angle_out1);
									angle_norm = fmaxf(angle_norm, 1.0e-4f);
									angle_norm2 = angle_norm * angle_norm;
									angle_dot = (angle_out0*angle_targ0 + angle_out1*angle_targ1) / angle_norm;
									angle_common = angle_weight*y_param.angle_sm[0]*angle_scale;
									delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] =
										angle_common*(angle_dot*angle_out0/angle_norm2 - angle_targ0/angle_norm);
									delta_o[(l_o+8+nb_class+nb_param+k+1)*f_offset] =
										angle_common*(angle_dot*angle_out1/angle_norm2 - angle_targ1/angle_norm);
									if(angle_unit_norm_scale > 0.0f)
									{
										angle_norm_diff = angle_norm - 1.0f;
										angle_norm_grad = angle_unit_norm_scale*angle_norm_diff*(angle_out0/angle_norm);
										delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] += angle_norm_grad;
										angle_norm_grad = angle_unit_norm_scale*angle_norm_diff*(angle_out1/angle_norm);
										delta_o[(l_o+8+nb_class+nb_param+k+1)*f_offset] += angle_norm_grad;
									}
								}
							}
							else
							{
								for(k = 0; k < nb_angle; k++)
									delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] =
										(angle_weight*y_param.angle_sm[0]*angle_scale
										*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k]));
								if(nb_angle == 2 && angle_unit_norm_scale > 0.0f)
								{
									angle_norm = sqrtf(
										(float)output[(l_o+8+nb_class+nb_param+0)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+0)*f_offset]
										+(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]);
									angle_norm = fmaxf(angle_norm, 1.0e-8f);
									angle_norm_diff = angle_norm - 1.0f;
									for(k = 0; k < nb_angle; k++)
									{
										angle_norm_grad = angle_unit_norm_scale*angle_norm_diff
											*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]/angle_norm);
										delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] += angle_norm_grad;
									}
								}
							}
					}
					else
						for(k = 0; k < nb_angle; k++)
							delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] = (0.0f);
					break;
				case 0:
					for(k = 0; k < nb_angle; k++)
						delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] =
							(y_param.angle_sm[0]*angle_scale*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-0.0f));
					break;
				case -1:
					for(k = 0; k < nb_angle; k++)
						delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] = (0.0f);
					break;
				}
				if(obb_loss_mode > 0 && obb_loss_scale > 0.0f && nb_angle >= 2
					&& max_IoU > min_angle_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))
				{
					obb_pred_theta = 0.5f*atan2f((float)output[(l_o+8+nb_class+nb_param+1)*f_offset], (float)output[(l_o+8+nb_class+nb_param+0)*f_offset]);
					obb_targ_theta = 0.5f*atan2f((float)target[l_t+7+nb_param+1], (float)target[l_t+7+nb_param+0]);
					obb_targ_cx = 0.5f*(targ_int[0] + targ_int[3]);
					obb_targ_cy = 0.5f*(targ_int[1] + targ_int[4]);
					yolo_obb_cov_loss_terms(c_box_in_pix[0], c_box_in_pix[1], c_box_in_pix[3], c_box_in_pix[4], obb_pred_theta, obb_targ_cx, obb_targ_cy, targ_size[0], targ_size[1], obb_targ_theta, &obb_grad_logw, &obb_grad_logh, &obb_grad_theta);
					delta_o[(l_o+0)*f_offset] += sm_tab[0][0]*coord_scale*obb_loss_scale*(float)output[(l_o+0)*f_offset]*(1.0f-(float)output[(l_o+0)*f_offset])*((float)output[(l_o+0)*f_offset]-obj_in_offset[0]);
					delta_o[(l_o+1)*f_offset] += sm_tab[0][0]*coord_scale*obb_loss_scale*(float)output[(l_o+1)*f_offset]*(1.0f-(float)output[(l_o+1)*f_offset])*((float)output[(l_o+1)*f_offset]-obj_in_offset[1]);
					delta_o[(l_o+3)*f_offset] += sm_tab[1][0]*size_scale*obb_loss_scale*obb_grad_logw;
					delta_o[(l_o+4)*f_offset] += sm_tab[1][0]*size_scale*obb_loss_scale*obb_grad_logh;
					obb_theta_norm2 = fmaxf((float)output[(l_o+8+nb_class+nb_param+0)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+0)*f_offset] +(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+1)*f_offset], 1.0e-6f);
					delta_o[(l_o+8+nb_class+nb_param+0)*f_offset] -= y_param.angle_sm[0]*obb_loss_scale*0.5f*obb_grad_theta*(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]/obb_theta_norm2;
					delta_o[(l_o+8+nb_class+nb_param+1)*f_offset] += y_param.angle_sm[0]*obb_loss_scale*0.5f*obb_grad_theta*(float)output[(l_o+8+nb_class+nb_param+0)*f_offset]/obb_theta_norm2;
				}
				if(multi_pos_topk > 1 && nb_im_iter > rand_startup && class_only_IoU < -1.9f)
				{
					multi_pos_iou_thr = fmaxf(multi_pos_min_iou, max_IoU*multi_pos_iou_ratio);
					for(secondary_iter = 1; secondary_iter < multi_pos_topk; secondary_iter++)
					{
						secondary_box = -1;
						best_secondary_iou = multi_pos_iou_thr;
						for(k = 0; k < nb_box; k++)
						{
							if(box_locked[k] == 2 || dist_prior[resp_targ_offset + k] >= -1.0f)
								continue;
							secondary_iou = IoU_table[resp_targ_offset + k];
							if(secondary_iou > best_secondary_iou)
							{
								best_secondary_iou = secondary_iou;
								secondary_box = k;
							}
						}
						if(secondary_box < 0)
							break;

						secondary_iou = best_secondary_iou;
						if(secondary_iou > 0.98f)
							secondary_iou = 0.98f;
						l_o = secondary_box*output_offset;

						for(k = 0; k < 6; k++)
							delta_o[(l_o+k)*f_offset] = 0.0f;
						for(k = 0; k < nb_class; k++)
							delta_o[(l_o+8+k)*f_offset] = 0.0f;
						for(k = 0; k < nb_param; k++)
							delta_o[(l_o+8+nb_class+k)*f_offset] = 0.0f;
						for(k = 0; k < nb_angle; k++)
							delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] = 0.0f;

						switch(fit_prob)
						{
							case 1:
								if(secondary_iou > min_prob_IoU_lim)
								{
										prob_target = yolo_probability_quality_target(&y_param, output, target,
											l_o, l_t, f_offset, nb_class, nb_param, nb_angle, secondary_iou, obj_in_offset);
									delta_o[(l_o+6)*f_offset] = multi_pos_obj_weight*(sm_tab[2][0]
										*prob_scale*(float)output[(l_o+6)*f_offset]
										*(1.0f-(float)output[(l_o+6)*f_offset])
										*((float)output[(l_o+6)*f_offset]-prob_target));
								}
								break;
							case 0:
								delta_o[(l_o+6)*f_offset] = multi_pos_obj_weight*(sm_tab[2][0]
									*prob_scale*(float)output[(l_o+6)*f_offset]
									*(1.0f-(float)output[(l_o+6)*f_offset])
									*((float)output[(l_o+6)*f_offset]-0.5f));
								break;
						}
						switch(fit_obj)
						{
							case 1:
								if(secondary_iou > min_obj_IoU_lim)
								{
									obj_target = yolo_objectness_quality_target(&y_param, output, target,
										l_o, l_t, f_offset, nb_class, nb_param, secondary_iou, obj_in_offset);
									delta_o[(l_o+7)*f_offset] = multi_pos_obj_weight*(sm_tab[3][0]
										*obj_scale*(float)output[(l_o+7)*f_offset]
										*(1.0f-(float)output[(l_o+7)*f_offset])
										*((float)output[(l_o+7)*f_offset]-obj_target));
								}
								break;
							case 0:
								delta_o[(l_o+7)*f_offset] = multi_pos_obj_weight*(sm_tab[3][0]
									*obj_scale*(float)output[(l_o+7)*f_offset]
									*(1.0f-(float)output[(l_o+7)*f_offset])
									*((float)output[(l_o+7)*f_offset]-0.5f));
								break;
						}
						box_locked[secondary_box] = 2;
						for(j = 0; j < nb_in_cell; j++)
							IoU_table[j*nb_box + secondary_box] = -2.0f;
					}
				}
				for(k = 0; k < nb_box; k++)
					IoU_table[resp_targ_offset + k] = -2.0f;
			}
		for(j = 0; j < nb_box; j++)
		{
			/* If no match only update Objectness toward 0 */
			/* (here it means error compute)! (no coordinate nor class update) */
			l_o = j*output_offset;
			if(box_locked[j] != 2)
			{
				for(k = 0; k < 6; k++)
					delta_o[(l_o+k)*f_offset] = 0.0f;
		
				if(box_locked[j] == 1)
				{
					delta_o[(l_o+6)*f_offset] = 0.0f;
					delta_o[(l_o+7)*f_offset] = 0.0f;
				}
				else
				{
					switch(fit_prob)
					{
						case 1:
							delta_o[(l_o+6)*f_offset] = (
								 sm_tab[2][0]*(lambda_noobj_prior[j])
								*prob_scale*(float)output[(l_o+6)*f_offset]
								*(1.0f-(float)output[(l_o+6)*f_offset])
								*((float)output[(l_o+6)*f_offset]-y_param.prob_quality_floor));
							break;
						case 0:
							delta_o[(l_o+6)*f_offset] = (
								 sm_tab[2][0]*(lambda_noobj_prior[j])
								*prob_scale*(float)output[(l_o+6)*f_offset]
								*(1.0f-(float)output[(l_o+6)*f_offset])
								*((float)output[(l_o+6)*f_offset]-0.5f));
							break;
						case -1:
							delta_o[(l_o+6)*f_offset] = (0.0f);
							break;
					}
					switch(fit_obj)
					{
						case 1:
							delta_o[(l_o+7)*f_offset] = (
								 sm_tab[3][0]*(lambda_noobj_prior[j])
								*obj_scale*(float)output[(l_o+7)*f_offset]
								*(1.0f-(float)output[(l_o+7)*f_offset])
								*((float)output[(l_o+7)*f_offset]-0.02f));
							break;
						case 0:
							delta_o[(l_o+7)*f_offset] = (
								 sm_tab[3][0]*(lambda_noobj_prior[j])
								*obj_scale*(float)output[(l_o+7)*f_offset]
								*(1.0f-(float)output[(l_o+7)*f_offset])
								*((float)output[(l_o+7)*f_offset]-0.5f));
							break;
						case -1:
							delta_o[(l_o+7)*f_offset] = (0.0f);
							break;
					}
				}
		
				for(k = 0; k < nb_class; k++)
					delta_o[(l_o+8+k)*f_offset] = (0.0f);
		
				for(k = 0; k < nb_param; k++)
					delta_o[(l_o+8+nb_class+k)*f_offset] = (0.0f);
				for(k = 0; k < nb_angle; k++)
					delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] = (0.0f);
			}
		}
	}
	#ifdef OPEN_MP
	}
	#endif
}


void YOLO_error_fct
	(float *i_output_error, void *i_output, void *i_target, int flat_target_size, int flat_output_size,
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, int size)
{		
	float *t_output = (float*) i_output;
	float *t_target = (float*) i_target;
	
	/* Define many "shorts" for y_param content to enhance code redeability*/
	int nb_box                      = y_param.nb_box;
	int nb_class                    = y_param.nb_class;
	int nb_param                    = y_param.nb_param; 
	int nb_angle                    = y_param.nb_angle;
	int strict_box_size_association = y_param.strict_box_size_association;
	float min_prior_forced_scaling  = y_param.min_prior_forced_scaling;
	int fit_dim                     = y_param.fit_dim;
	int class_softmax               = y_param.class_softmax;
	int diff_flag                   = y_param.diff_flag;
	int error_type                  = y_param.error_type;
	int prior_dist_type             = y_param.prior_dist_type;
	int target_box_mode             = y_param.target_box_mode;
	int obb_loss_mode               = y_param.obb_loss_mode;

	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];
	float angle_scale = y_param.angle_scale, angle_unit_norm_scale = y_param.angle_unit_norm_scale;
	float obb_loss_scale = y_param.obb_loss_scale;

	float *prior_size         = y_param.prior_size;
	int   *cell_size          = y_param.cell_size;
	float *param_ind_scale    = y_param.param_ind_scale;
	float *lambda_noobj_prior = y_param.noobj_prob_prior;
	float **sm_tab            = y_param.slopes_and_maxes_tab;
	float *t_IoU_monitor      = y_param.IoU_monitor;
	int   *t_target_cell_mask = y_param.target_cell_mask;
	float *t_IoU_table        = y_param.IoU_table;
	float *t_dist_prior       = y_param.dist_prior;
	int   *t_box_locked       = y_param.box_locked;
	float *t_box_in_pix       = y_param.box_in_pix;
	
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);
	float good_IoU_lim      = y_param.IoU_limits[0], low_IoU_best_box_assoc = y_param.IoU_limits[1];
	float min_prob_IoU_lim  = y_param.IoU_limits[2], min_obj_IoU_lim        = y_param.IoU_limits[3];
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim      = y_param.IoU_limits[5];
	float min_angle_IoU_lim = y_param.min_angle_IoU_lim;
	float diff_IoU_lim      = y_param.IoU_limits[6], diff_obj_lim           = y_param.IoU_limits[7];
	int fit_pos = y_param.fit_parts[0], fit_size  = y_param.fit_parts[1], fit_prob  = y_param.fit_parts[2];
	int fit_obj = y_param.fit_parts[3], fit_class = y_param.fit_parts[4], fit_param = y_param.fit_parts[5];
	int fit_angle = y_param.fit_angle;
	int angle_loss_mode = y_param.angle_loss_mode;
	
	#pragma omp parallel
	#ifdef OPEN_MP
	{
	srand((int)time(NULL) ^ omp_get_thread_num());
	#endif
	#pragma omp for schedule(guided,4)
	for(int c_pix = 0; c_pix < size; c_pix++)
	{	
		float *output, *target, *output_error;
		int *target_cell_mask, *box_locked;
		float *IoU_table, *dist_prior, *box_in_pix, *IoU_monitor;
		int l_o, l_t, i, j, k, l;
		size_t f_offset, c_total_nb_area, c_total_nb_area_batch, total_cell_pos_nb_area, total_area_and_cell_offset;
		int c_batch, output_offset, target_offset, nb_obj_target, s_p_i = 0;
		int nb_in_cell, id_in_cell, id_in_cell_offset, resp_box = -1, resp_targ = -1, resp_targ_offset, targ_diff_flag = 0;
		float best_dist, c_dist, max_IoU, current_IoU, prob_target, obj_target;
		int cell_pos[3], c_nb_area[3], obj_c[3];
		float *c_box_in_pix, *c_prior_size;
			float obj_in_offset[6], out_int[6], targ_int[6], targ_size[3];
				float obb_grad_logw, obb_grad_logh, obb_grad_theta;
			float obb_pred_theta, obb_targ_theta, obb_targ_cx, obb_targ_cy, obb_loss_val;
			float angle_weight, angle_norm, angle_norm_diff, angle_unit_err_share;
			float angle_out0, angle_out1, angle_targ0, angle_targ1, angle_dot, angle_pair_loss;
			float class_only_IoU = -2.0f;
	
		c_nb_area[0] = nb_area_w; c_nb_area[1] = nb_area_h; c_nb_area[2] = nb_area_d;
		c_total_nb_area = c_nb_area[0]*c_nb_area[1]*c_nb_area[2];
		c_batch = c_pix / flat_output_size;
		target = t_target + flat_target_size * c_batch;
		f_offset = size;
		output_offset = 8+nb_class+nb_param+nb_angle;
		target_offset = 7+nb_param+((nb_angle > 0) ? nb_angle + 1 : 0)+diff_flag;
		
		i = c_pix % flat_output_size;
		cell_pos[2] = i / (c_nb_area[0]*c_nb_area[1]);
		cell_pos[1] = (int)(i % (c_nb_area[0]*c_nb_area[1])) / c_nb_area[0];
		cell_pos[0] = (int)(i % (c_nb_area[0]*c_nb_area[1])) % c_nb_area[0];
		
		c_total_nb_area_batch = c_total_nb_area * c_batch;
		total_cell_pos_nb_area = cell_pos[2]*c_nb_area[0]*c_nb_area[1] + cell_pos[1]*c_nb_area[0] + cell_pos[0];
		total_area_and_cell_offset = c_total_nb_area_batch + total_cell_pos_nb_area;
		
		output_error = i_output_error + total_area_and_cell_offset;
		output = t_output + total_area_and_cell_offset;
		
		IoU_monitor = t_IoU_monitor + 2 * nb_box * total_area_and_cell_offset;
		target_cell_mask = t_target_cell_mask + total_area_and_cell_offset * y_param.max_nb_obj_per_image;
		IoU_table  = t_IoU_table  + total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;
		dist_prior = t_dist_prior + total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;
		box_locked = t_box_locked + total_area_and_cell_offset * nb_box;
		box_in_pix = t_box_in_pix + total_area_and_cell_offset * 6 * nb_box;
		
		nb_obj_target = target[0];
		target++;
		
		if(nb_obj_target == -1)
		{
			nb_obj_target = 1;
			class_only_IoU = good_IoU_lim;
		}
		
		best_dist = 1000000000;
		for(k = 0; k < nb_box; k++)
		{
			box_locked[k] = 0;
			c_box_in_pix = box_in_pix + k*6;
			c_prior_size = prior_size + k*3;
			l_o = k*output_offset;
			for(l = 0; l < 3; l++)
				c_box_in_pix[l] = ((float)output[(l_o+l)*f_offset] + cell_pos[l]) * cell_size[l];
			for(l = 0; l < 3; l++)
				c_box_in_pix[l+3] = c_prior_size[l]*expf((float)output[(l_o+l+3)*f_offset]);
			
			c_dist = sqrt(c_prior_size[0]*c_prior_size[0]
						+ c_prior_size[1]*c_prior_size[1]
						+ c_prior_size[2]*c_prior_size[2]);
			if(c_dist < best_dist)
			{
				best_dist = c_dist;
				s_p_i = k;
			}
			
			IoU_monitor[k*2] = -1.0f;
			IoU_monitor[k*2+1] = -1.0f;
		}
		
		nb_in_cell = 0;
		for(j = 0; j < nb_obj_target; j++)
		{
			l_t = j*target_offset;
			yolo_load_target_box(target, l_t, targ_int, target_box_mode);
			
			/* Search for targets that should be predicted by the current cell element */
			target_cell_mask[j] = 1;
			for(l = 0; l < 3; l++)
			{
				obj_c[l] = (int)( (targ_int[l+3] + targ_int[l])*0.5f / cell_size[l]);
				/* If target outside the current cell element, set target flag to 0*/
				if(obj_c[l] != cell_pos[l])
					target_cell_mask[j] = 0;
			}
			
			if(target_cell_mask[j] == 1)
				nb_in_cell++;
			
			/* Flag all the "Good but not best boxes" for all targets regardless of the grid element */
			for(k = 0; k < nb_box; k++)
			{
				l_o = k*output_offset;
				if(box_locked[k] != 0)
					continue;
				c_box_in_pix = box_in_pix+k*6;
				for(l = 0; l < 6; l++)
					out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];
				
				current_IoU = yolo_box_quality(&y_param, out_int, targ_int, output, target,
					l_o, l_t, f_offset, nb_class, nb_param, nb_angle);
				if(current_IoU > good_IoU_lim)
					box_locked[k] = 1;
			}
		}
		
		id_in_cell = 0;
		for(j = 0; j < nb_obj_target; j++)
		{
			id_in_cell_offset = id_in_cell*nb_box;
				if(target_cell_mask[j] == 0)
				{
					continue;
				}
			
				l_t = j*target_offset;
					yolo_load_target_box(target, l_t, targ_int, target_box_mode);
				for(l = 0; l < 3; l++)
				{
					targ_size[l] = targ_int[l+3] - targ_int[l];
				}
				
				for(k = 0; k < nb_box; k++)
				{
					l_o = k*output_offset;
					c_box_in_pix = box_in_pix+k*6;
					for(l = 0; l < 6; l++)
						out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];
					
					current_IoU = yolo_box_quality(&y_param, out_int, targ_int, output, target,
						l_o, l_t, f_offset, nb_class, nb_param, nb_angle);
					IoU_table[id_in_cell_offset + k] = current_IoU;
					dist_prior[id_in_cell_offset + k] = -2.0f;
				}
			
			/* Restrict the association to the l best theoritical prior (times repetition of identical priors) */
			if(error_type == ERR_COMPLETE && strict_box_size_association > 0)
			{
				if(prior_dist_type == DIST_IOU)
					for(l = 0; l < 6; l++)
						targ_int[l] = copysignf(0.5f,l-2.5f)*targ_size[l%3];
				
				for(k = 0; k < nb_box; k++)
				{
					c_prior_size = prior_size + k*3;
					switch(prior_dist_type)
					{
						case DIST_IOU:
							for(l = 0; l < 6; l++)
								out_int[l] = copysignf(0.5f,l-2.5f)*c_prior_size[l%3];
							dist_prior[id_in_cell_offset + k] = 1.0f - y_param.c_IoU_fct(out_int, targ_int);
							break;
						
						default:
						case DIST_SIZE:
							dist_prior[id_in_cell_offset + k] = sqrt(
								 (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])
								+(targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])
								+(targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]));
							break;
						
						case DIST_OFFSET:
							for(l = 0; l < 3; l++)
							{
								obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];
								if(obj_in_offset[l+3] < size_min_sat)
									obj_in_offset[l+3] = logf(size_min_sat);
								else if(obj_in_offset[l+3] > size_max_sat)
									obj_in_offset[l+3] = logf(size_max_sat);
								else
									obj_in_offset[l+3] = logf(obj_in_offset[l+3]);
							}
							
							dist_prior[id_in_cell_offset + k] = 
								 fabs(obj_in_offset[3])
								+fabs(obj_in_offset[4])
								+fabs(obj_in_offset[5]);
							break;
					}
				}
				
				for(l = 0; l < strict_box_size_association; l++)
				{
					best_dist = 1000000.0f;
					for(k = 0; k < nb_box; k++)
						if(dist_prior[id_in_cell_offset+k] > 0.0 && dist_prior[id_in_cell_offset+k] < best_dist)
							best_dist = dist_prior[id_in_cell_offset+k];
					for(k = 0; k < nb_box; k++) /* Flag the closest theoritical prior (and identical ones if any) */
						if(fabs(dist_prior[id_in_cell_offset+k] - best_dist) < 0.001f )
							dist_prior[id_in_cell_offset+k] = -2.0f;
				}
			}
			
			id_in_cell++;
		}
		
		for(id_in_cell = 0; id_in_cell < nb_in_cell; id_in_cell++)
		{
			/* No random association in error display*/
			max_IoU = -2.0f; resp_box = -1;	resp_targ = -1;
			for(j = 0; j < nb_in_cell; j++)
				for(k = 0; k < nb_box; k++)
					if(IoU_table[j*nb_box+k] > max_IoU && dist_prior[j*nb_box+k] < -1.0f)
					{
						max_IoU = IoU_table[j*nb_box+k];
						resp_targ = j;
						resp_box = k;
					}
			
			/* If strict_box_size > 0 and no more good prior is available, or if there is more targets than boxes */
			/* In that case all the remaining target are unable to be associated to */
			/* any other box and the id_in_cell loop must be stoped */
			if(resp_box == -1)
				continue;
			
			/* l is the "best" index in the "in cell" list */
			/*Need to get back the original target index from the "in cell" index*/
			l = 0;
			for(j = 0; j < nb_obj_target; j++)
			{
				l += target_cell_mask[j];
				if(l == resp_targ + 1)
					break;
			}
			/* The appropriate j is defined after this early stop loop*/
			l_t = j*target_offset;
			resp_targ_offset = resp_targ*nb_box;
			
			if(error_type == ERR_COMPLETE)
			{
			yolo_load_target_box(target, l_t, targ_int, target_box_mode);
				for(l = 0; l < 3; l++)
					targ_size[l] = targ_int[l+3] - targ_int[l];
				
				/* Force the association to the smallest prior (or identical) if the target is too small */
				if(targ_size[0] < min_prior_forced_scaling*prior_size[s_p_i*3+0]
					&& targ_size[1] < min_prior_forced_scaling*prior_size[s_p_i*3+1]
					&& targ_size[2] < min_prior_forced_scaling*prior_size[s_p_i*3+2])
				{
					max_IoU = -2.0f; best_dist = prior_size[s_p_i*3+0]*prior_size[s_p_i*3+1]*prior_size[s_p_i*3+2];
						for(k = 0; k < nb_box; k++)
						{
							c_prior_size = prior_size + k*3;
							if((prior_size[s_p_i*3+0] == c_prior_size[0]
								&& prior_size[s_p_i*3+1] == c_prior_size[1]
								&& prior_size[s_p_i*3+2] == c_prior_size[2])
								&& IoU_table[resp_targ_offset+k] > max_IoU)
							{
								max_IoU = IoU_table[resp_targ_offset+k];
								resp_box = k;
						}
					}
				}
				/* If prediction is too bad, associate it it the best theoritical prior instead (might found the same box again) */
				/* Also force the best theoritical prior association at a small rate */
				else if(max_IoU < low_IoU_best_box_assoc)
				{
					if(prior_dist_type == DIST_IOU)
						for(l = 0; l < 6; l++)
							targ_int[l] = copysignf(0.5f,l-2.5f)*targ_size[l%3];
					
					best_dist = 100000.0f;
					for(k = 0; k < nb_box; k++)
					{
						c_prior_size = prior_size + k*3;
						switch(prior_dist_type)
						{
							case DIST_IOU:
								for(l = 0; l < 6; l++)
									out_int[l] = copysignf(0.5f,l-2.5f)*c_prior_size[l%3];
								dist_prior[resp_targ_offset + k] = 1.0f - y_param.c_IoU_fct(out_int, targ_int);
								break;
							
							default:
							case DIST_SIZE:
								dist_prior[resp_targ_offset + k] = sqrt(
									 (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])
									+(targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])
									+(targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]));
								break;
							
							case DIST_OFFSET:
								for(l = 0; l < 3; l++)
								{
									obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];
									if(obj_in_offset[l+3] < size_min_sat)
										obj_in_offset[l+3] = logf(size_min_sat);
									else if(obj_in_offset[l+3] > size_max_sat)
										obj_in_offset[l+3] = logf(size_max_sat);
									else
										obj_in_offset[l+3] = logf(obj_in_offset[l+3]);
								}
								
								dist_prior[resp_targ_offset + k] = 
									 fabs(obj_in_offset[3])
									+fabs(obj_in_offset[4])
									+fabs(obj_in_offset[5]);
								break;
						}
						if(dist_prior[resp_targ_offset + k] < best_dist)
							best_dist = dist_prior[resp_targ_offset + k];
					}
					max_IoU = -2.0f;
					for(k = 0; k < nb_box; k++)
					{
						if(fabsf(dist_prior[resp_targ_offset + k] - best_dist) < 0.001f && IoU_table[resp_targ_offset + k] > max_IoU)
						{
							max_IoU = IoU_table[resp_targ_offset + k];
							resp_box = k;
						}
					}
					/* If the best prior (or identical) is not available, the resp_box is unchanged */
					/* Should always get a resp_box != -1, regarding all previous conditions */
				}
			}
			
			/* Mark the target as already associated by removing its contributions to the IoU table */
				for(k = 0; k < nb_box; k++)
					IoU_table[resp_targ_offset + k] = -2.0f;
				
				c_box_in_pix = box_in_pix + resp_box*6;
				for(l = 0; l < 6; l++)
					out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];
				
			yolo_load_target_box(target, l_t, targ_int, target_box_mode);
				for(l = 0; l < 3; l++)
					targ_size[l] = targ_int[l+3] - targ_int[l];
				
				l_o = resp_box*output_offset;
				max_IoU = yolo_box_quality(&y_param, out_int, targ_int, output, target,
					l_o, l_t, f_offset, nb_class, nb_param, nb_angle);
				if(max_IoU > 0.98f)
					max_IoU = 0.98f;
				if(class_only_IoU > -2.0f)
					max_IoU = class_only_IoU; /*regardless of actual IoU because class only box is not precise*/
				
				c_prior_size = prior_size + 3*resp_box;
			
			/* Positive reinforcement */
			targ_diff_flag = 0;
			if(diff_flag)	/* Cast from mixed precision type to float is always possible, but not necessary to int directly */
				targ_diff_flag = (int)((float)target[l_t+7+nb_param+((nb_angle > 0) ? nb_angle + 1 : 0)]);
			
			/* If the target is flagged as "difficult", only update the matching box if the prediction is already confident enough */
			/* The target is removed from the list anyway, and the corresponding box fall to "background" or "Good_but_not_best" case*/
			if(diff_flag && targ_diff_flag > 0
				&& (error_type == ERR_NATURAL || max_IoU < diff_IoU_lim || (float)output[(l_o+7)*f_offset] < diff_obj_lim))
				continue;
			
			/* Mark the box as already associated by removing its contributions to the IoU table */
			for(j = 0; j < nb_in_cell; j++)
				IoU_table[j*nb_box + resp_box] = -2.0f;
			
			box_locked[resp_box] = 2;
			
			IoU_monitor[resp_box*2] = (float)output[(l_o+7)*f_offset];
			IoU_monitor[resp_box*2+1] = max_IoU;
			
			for(l = 0; l < 3; l++)
				obj_in_offset[l] = ((targ_int[l+3] + targ_int[l])*0.5f - cell_pos[l]*cell_size[l])/(float)cell_size[l];
			for(l = 0; l < 3; l++)
			{
				obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];
				if(obj_in_offset[l+3] < size_min_sat)
					obj_in_offset[l+3] = logf(size_min_sat);
				else if(obj_in_offset[l+3] > size_max_sat)
					obj_in_offset[l+3] = logf(size_max_sat);
				else
					obj_in_offset[l+3] = logf(obj_in_offset[l+3]);
			}
			
			switch(fit_pos)
			{
				case 1:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k && class_only_IoU < -1.9f && (diff_flag == 0 || targ_diff_flag < 3))
							output_error[(l_o+k)*f_offset] = 0.5f*coord_scale
								*((float)output[(l_o+k)*f_offset]-obj_in_offset[k])
								*((float)output[(l_o+k)*f_offset]-obj_in_offset[k]);
						else
							output_error[(l_o+k)*f_offset] = 0.0f;
					}
					break;
				case 0:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k)
							output_error[(l_o+k)*f_offset] = 0.5f*coord_scale
								*((float)output[(l_o+k)*f_offset]-0.5f)
								*((float)output[(l_o+k)*f_offset]-0.5f);
						else
							output_error[(l_o+k)*f_offset] = 0.0f;
					}
					break;
				case -1:
					for(k = 0; k < 3; k++)
						output_error[(l_o+k)*f_offset] = 0.0f;
					break;
			}
		
			switch(fit_size)
			{
				case 1:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k && class_only_IoU < -1.9f && (diff_flag == 0 || targ_diff_flag < 3))
							output_error[(l_o+k+3)*f_offset] = 0.5f*size_scale
							*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3])
							*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3]);
						else
							output_error[(l_o+k+3)*f_offset] = 0.0f;
					}
					break;
				case 0:
					for(k = 0; k < 3; k++)
					{
						if(fit_dim > k)
							output_error[(l_o+k+3)*f_offset] = 0.5f*size_scale
							*((float)output[(l_o+k+3)*f_offset]-0.0f)
							*((float)output[(l_o+k+3)*f_offset]-0.0f);
						else
							output_error[(l_o+k+3)*f_offset] = 0.0f;
					}
					break;
				case -1:
					for(k = 0; k < 3; k++)
						output_error[(l_o+k+3)*f_offset] = 0.0f;
					break;
			}
		
			switch(fit_prob)
			{
				case 1:
					if(max_IoU > min_prob_IoU_lim || error_type == ERR_NATURAL)
					{
							prob_target = yolo_probability_quality_target(&y_param, output, target,
								l_o, l_t, f_offset, nb_class, nb_param, nb_angle, max_IoU, obj_in_offset);
						output_error[(l_o+6)*f_offset] = 0.5f*prob_scale
							*((float)output[(l_o+6)*f_offset]-prob_target)
							*((float)output[(l_o+6)*f_offset]-prob_target);
					}
					else
						output_error[(l_o+6)*f_offset] = 0.0f;
					break;
				case 0:
					output_error[(l_o+6)*f_offset] = 0.5f*prob_scale
						*((float)output[(l_o+6)*f_offset]-0.5f)
						*((float)output[(l_o+6)*f_offset]-0.5f);
					break;
				case -1:
					output_error[(l_o+6)*f_offset] = 0.0f;
					break;
			}
		
			switch(fit_obj)
			{
				case 1:
					if(max_IoU > min_obj_IoU_lim || error_type == ERR_NATURAL)
					{
						obj_target = yolo_objectness_quality_target(&y_param, output, target,
							l_o, l_t, f_offset, nb_class, nb_param, max_IoU, obj_in_offset);
						output_error[(l_o+7)*f_offset] = 0.5f*obj_scale
							*((float)output[(l_o+7)*f_offset]-obj_target)
							*((float)output[(l_o+7)*f_offset]-obj_target);
					}
					else
						output_error[(l_o+7)*f_offset] = 0.0f;
					break;
				case 0:
					output_error[(l_o+7)*f_offset] = 0.5f*obj_scale
						*((float)output[(l_o+7)*f_offset]-0.5)
						*((float)output[(l_o+7)*f_offset]-0.5);
					break;
				case -1:
					output_error[(l_o+7)*f_offset] = 0.0f;
					break;
			}
		
			/*Note : mean square error on classes => could be changed to soft max but difficult to balance*/
			switch(fit_class)
			{
				case 1:
					if((max_IoU > min_class_IoU_lim && (diff_flag == 0 || targ_diff_flag < 3)) || error_type == ERR_NATURAL)
					{
						if(class_softmax)
						{
							for(k = 0; k < nb_class; k++)
							{
								if(k == (int)target[l_t]-1)
								{
									if((float)output[(l_o+8+k)*f_offset] > 0.0000001f)
										output_error[(l_o+8+k)*f_offset] = class_scale
											*(-logf((float)output[(l_o+8+k)*f_offset]));
									else
										output_error[(l_o+8+k)*f_offset] = class_scale*(-logf(0.0000001f));
								}
								else
									output_error[(l_o+8+k)*f_offset] = 0.0f;
							}
						}
						else
						{
							for(k = 0; k < nb_class; k++)
							{
								if(k == (int)target[l_t]-1)
									output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale
										*((float)output[(l_o+8+k)*f_offset]-0.98f)
										*((float)output[(l_o+8+k)*f_offset]-0.98f);
								else
									output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale
										*((float)output[(l_o+8+k)*f_offset]-0.02f)
										*((float)output[(l_o+8+k)*f_offset]-0.02f);
							}
						}
					}
					else
						for(k = 0; k < nb_class; k++)
							output_error[(l_o+8+k)*f_offset] = 0.0f;
					break;
				case 0:
					if(class_softmax)
					{
						/* Could compute CE with target = 1/nb_class, but in this case perfect classification error > 0 (still minimum) */
						for(k = 0; k < nb_class; k++)
							output_error[(l_o+8+k)*f_offset] = 0.0f;
					}
					else
					{
						for(k = 0; k < nb_class; k++)
							output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale
								*((float)output[(l_o+8+k)*f_offset]-0.5f)
								*((float)output[(l_o+8+k)*f_offset]-0.5f);
					}
					break;
				case -1:
					for(k = 0; k < nb_class; k++)
						output_error[(l_o+8+k)*f_offset] = 0.0f;
					break;
			}
		
			/*Linear error of additional parameters*/
			switch(fit_param)
			{
				case 1:
					if((max_IoU > min_param_IoU_lim && (diff_flag == 0 || targ_diff_flag < 3)) || error_type == ERR_NATURAL)
						for(k = 0; k < nb_param; k++)
						{
							if(yolo_flux_refine_is_aux_param(&y_param, k, nb_param))
							{
								output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;
								continue;
							}
							output_error[(l_o+8+nb_class+k)*f_offset] = (param_ind_scale[k]*0.5f*param_scale
								*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k])
								*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k]));
						}
					else
						for(k = 0; k < nb_param; k++)
							output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;
					break;
				case 0:
					for(k = 0; k < nb_param; k++)
					{
						if(yolo_flux_refine_is_aux_param(&y_param, k, nb_param))
						{
							output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;
							continue;
						}
						output_error[(l_o+8+nb_class+k)*f_offset] = (param_ind_scale[k]*0.5f*param_scale
							*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f)
							*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f));
					}
					break;
				case -1:
					for(k = 0; k < nb_param; k++)
						output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;
					break;
			}
			
			/*Encoded angle head display error*/
			switch(fit_angle)
			{
				case 1:
						if(nb_angle > 0 && ((max_IoU > min_angle_IoU_lim && (diff_flag == 0 || targ_diff_flag < 3)) || error_type == ERR_NATURAL))
						{
							angle_weight = (float)target[l_t+7+nb_param+nb_angle];
							if(angle_loss_mode == 1)
							{
								for(k = 0; k < nb_angle; k += 2)
								{
									if(k + 1 >= nb_angle)
									{
										output_error[(l_o+8+nb_class+nb_param+k)*f_offset] =
											(angle_weight*0.5f*angle_scale
											*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k])
											*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k]));
										continue;
									}
									angle_out0 = (float)output[(l_o+8+nb_class+nb_param+k)*f_offset];
									angle_out1 = (float)output[(l_o+8+nb_class+nb_param+k+1)*f_offset];
									angle_targ0 = (float)target[l_t+7+nb_param+k];
									angle_targ1 = (float)target[l_t+7+nb_param+k+1];
									angle_norm = sqrtf(angle_out0*angle_out0 + angle_out1*angle_out1);
									angle_norm = fmaxf(angle_norm, 1.0e-8f);
									angle_dot = (angle_out0*angle_targ0 + angle_out1*angle_targ1) / angle_norm;
									angle_dot = fminf(1.0f, fmaxf(-1.0f, angle_dot));
									angle_pair_loss = angle_weight*angle_scale*(1.0f - angle_dot);
									angle_unit_err_share = 0.0f;
									if(angle_unit_norm_scale > 0.0f)
									{
										angle_norm_diff = angle_norm - 1.0f;
										angle_unit_err_share = 0.25f*angle_unit_norm_scale*angle_norm_diff*angle_norm_diff;
									}
									output_error[(l_o+8+nb_class+nb_param+k)*f_offset] = 0.5f*angle_pair_loss + angle_unit_err_share;
									output_error[(l_o+8+nb_class+nb_param+k+1)*f_offset] = 0.5f*angle_pair_loss + angle_unit_err_share;
								}
							}
							else
							{
								angle_unit_err_share = 0.0f;
								if(nb_angle == 2 && angle_unit_norm_scale > 0.0f)
								{
									angle_norm = sqrtf(
										(float)output[(l_o+8+nb_class+nb_param+0)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+0)*f_offset]
										+(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]);
									angle_norm_diff = angle_norm - 1.0f;
									angle_unit_err_share = 0.25f*angle_unit_norm_scale*angle_norm_diff*angle_norm_diff;
								}
								for(k = 0; k < nb_angle; k++)
									output_error[(l_o+8+nb_class+nb_param+k)*f_offset] =
										(angle_weight*0.5f*angle_scale
										*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k])
										*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k])
										+angle_unit_err_share);
							}
						}
					else
						for(k = 0; k < nb_angle; k++)
							output_error[(l_o+8+nb_class+nb_param+k)*f_offset] = 0.0f;
					break;
				case 0:
					for(k = 0; k < nb_angle; k++)
						output_error[(l_o+8+nb_class+nb_param+k)*f_offset] = 0.5f*angle_scale
							*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset])
							*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]);
					break;
				case -1:
					for(k = 0; k < nb_angle; k++)
						output_error[(l_o+8+nb_class+nb_param+k)*f_offset] = 0.0f;
					break;
				}
				if(obb_loss_mode > 0 && obb_loss_scale > 0.0f && nb_angle >= 2
					&& ((max_IoU > min_angle_IoU_lim && (diff_flag == 0 || targ_diff_flag < 3)) || error_type == ERR_NATURAL))
				{
					obb_pred_theta = 0.5f*atan2f((float)output[(l_o+8+nb_class+nb_param+1)*f_offset], (float)output[(l_o+8+nb_class+nb_param+0)*f_offset]);
					obb_targ_theta = 0.5f*atan2f((float)target[l_t+7+nb_param+1], (float)target[l_t+7+nb_param+0]);
					obb_targ_cx = 0.5f*(targ_int[0] + targ_int[3]);
					obb_targ_cy = 0.5f*(targ_int[1] + targ_int[4]);
					obb_loss_val = 0.5f*obb_loss_scale*yolo_obb_cov_loss_terms(c_box_in_pix[0], c_box_in_pix[1], c_box_in_pix[3], c_box_in_pix[4], obb_pred_theta, obb_targ_cx, obb_targ_cy, targ_size[0], targ_size[1], obb_targ_theta, &obb_grad_logw, &obb_grad_logh, &obb_grad_theta);
					output_error[(l_o+0)*f_offset] += 0.25f*obb_loss_val;
					output_error[(l_o+1)*f_offset] += 0.25f*obb_loss_val;
					output_error[(l_o+3)*f_offset] += 0.25f*obb_loss_val;
					output_error[(l_o+4)*f_offset] += 0.25f*obb_loss_val;
					output_error[(l_o+8+nb_class+nb_param+0)*f_offset] += 0.25f*obb_loss_val;
					output_error[(l_o+8+nb_class+nb_param+1)*f_offset] += 0.25f*obb_loss_val;
				}
			}
		for(j = 0; j < nb_box; j++)
		{
			/*If no match only update Objectness toward 0 */
			/*(here it means error compute)! (no coordinate nor class update)*/
			l_o = j*output_offset;
			if(box_locked[j] != 2)
			{
				for(k = 0; k < 6; k++)
					output_error[(l_o+k)*f_offset] = 0.0f;
		
				if(box_locked[j] == 1)
				{
					output_error[(l_o+6)*f_offset] = 0.0f;
					output_error[(l_o+7)*f_offset] = 0.0f;
				}
				else
				{
					switch(fit_prob)
					{
						case 1:
							output_error[(l_o+6)*f_offset] = 0.5f*(lambda_noobj_prior[j])*prob_scale
								*((float)output[(l_o+6)*f_offset]-y_param.prob_quality_floor)
								*((float)output[(l_o+6)*f_offset]-y_param.prob_quality_floor);
							break;
						case 0:
							output_error[(l_o+6)*f_offset] = 0.5f*(lambda_noobj_prior[j])*prob_scale
								*((float)output[(l_o+6)*f_offset]-0.5f)
								*((float)output[(l_o+6)*f_offset]-0.5f);
							break;
						case -1:
							output_error[(l_o+6)*f_offset] = 0.0f;
							break;
					}
		
					switch(fit_obj)
					{
						case 1:
							output_error[(l_o+7)*f_offset] = 0.5f*(lambda_noobj_prior[j])*obj_scale
								*((float)output[(l_o+7)*f_offset]-0.02f)
								*((float)output[(l_o+7)*f_offset]-0.02f);
							break;
						case 0:
							output_error[(l_o+7)*f_offset] = 0.5f*(lambda_noobj_prior[j])*obj_scale
								*((float)output[(l_o+7)*f_offset]-0.5f)
								*((float)output[(l_o+7)*f_offset]-0.5f);
							break;
						case -1:
							output_error[(l_o+7)*f_offset] = 0.0f;
							break;
					}
				}
				for(k = 0; k < nb_class; k++)
					output_error[(l_o+8+k)*f_offset] = 0.0f;
				for(k = 0; k < nb_param; k++)
					output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;
				for(k = 0; k < nb_angle; k++)
					output_error[(l_o+8+nb_class+nb_param+k)*f_offset] = 0.0f;
			}
		}
	}
	#ifdef OPEN_MP
	}
	#endif
}


void YOLO_activation(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	
	YOLO_activation_fct(current->output, (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) 
		* current->c_network->batch_size, a_param->biased_dim*current->c_network->length, *a_param, a_param->size, a_param->class_softmax);
}


void YOLO_deriv(layer *previous)
{
	printf("Error : YOLO activation can not be used in the middle of the network !\n");
	exit(EXIT_FAILURE);
}


void YOLO_deriv_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	
	YOLO_deriv_error_fct(current->delta_o, current->output, current->c_network->target, current->c_network->output_dim, 
		(size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]), c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], 
		*a_param, (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) * current->c_network->batch_size,
		current->c_network->iter * current->c_network->train.size);
}


void YOLO_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	
	YOLO_error_fct((float*)current->c_network->output_error, current->output, current->c_network->target, current->c_network->output_dim, 
		(size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]), c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], 
		*a_param, (size_t)(c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2]) * current->c_network->batch_size);
}



//#####################################################
