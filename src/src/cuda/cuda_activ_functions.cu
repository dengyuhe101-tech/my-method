	
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


#include "../prototypes.h"

// Local variables
static int cu_blocks;

// Public are in "prototypes.h"

// Private prototypes
__device__ float gpu_IoU_fct(float *output, float *target);
__device__ float gpu_GIoU_fct(float *output, float *target);
__device__ float gpu_DIoU_fct(float *output, float *target);
__device__ float gpu_DIoU2_fct(float *output, float *target);
void cuda_linear_activation(layer *current);
void cuda_linear_deriv(layer *previous);
void cuda_linear_deriv_output_error(layer *current);
void cuda_linear_output_error(layer *current);
void cuda_ReLU_activation(layer *current);
void cuda_ReLU_deriv(layer *previous);
void cuda_ReLU_deriv_output_error(layer* current);
void cuda_ReLU_output_error(layer* current);
void cuda_logistic_activation(layer *current);
void cuda_logistic_deriv(layer *previous);
void cuda_logistic_deriv_output_error(layer* current);
void cuda_logistic_output_error(layer* current);
void cuda_softmax_activation(layer *current);
void cuda_softmax_deriv(layer *previous);
void cuda_softmax_deriv_output_error(layer *current);
void cuda_softmax_output_error(layer *current);
void cuda_YOLO_activation(layer *current);
void cuda_YOLO_deriv(layer *previous);
void cuda_YOLO_deriv_output_error(layer *current);
void cuda_YOLO_output_error(layer *current);
void cuda_YOLO_activ_init(layer *current);

// Functions that result from templates are not listed here but at the end of the file instead

static void cuda_check_yolo_kernel(const char *kernel_name)
{
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("\n CUDA ERROR after %s launch: %s\n", kernel_name, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaDeviceSynchronize();
	if(err != cudaSuccess)
	{
		printf("\n CUDA ERROR while synchronizing %s: %s\n", kernel_name, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

//Is in fact a leaky ReLU, to obtain true ReLU set leaking_factor to 0
#define linear_activation_kernel(name, type)																									\
__global__ void linear_activation_kernel_##name(void *i_tab, int dim, int biased_dim, int offset, int length, size_t size)						\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* tab = (type*) i_tab;																													\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i >= (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
			tab[i] = (type) 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset >= length)																											\
			tab[i] = (type) 0.0f;																												\
	}																																			\
}

#define linear_deriv_kernel(name, type)																											\
__global__ void linear_deriv_kernel_##name(void *i_deriv, int dim, int biased_dim, int offset, int length, size_t size)							\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* deriv = (type*) i_deriv;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i >= (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset >= length)																											\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
}


//#####################################################
//		  ReLU activation related templates
//#####################################################

//Is in fact a leaky ReLU, to obtain true ReLU set leaking_factor to 0
#define ReLU_activation_kernel(name, type)																										\
__global__ void ReLU_activation_kernel_##name(void *i_tab, int dim, int biased_dim, int offset,													\
	float saturation, float leaking_factor, int length, size_t size)																			\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* tab = (type*) i_tab;																													\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			if(tab[i] <= (type) 0.0f)																											\
				tab[i] *= (type) leaking_factor;																								\
			else if(tab[i] > (type) saturation)																									\
				tab[i] = (type) saturation + (tab[i] - (type) saturation)*((type)leaking_factor);												\
		}																																		\
		else																																	\
			tab[i] = (type) 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			if(tab[i] <= (type) 0.0f)																											\
				tab[i] *= (type) leaking_factor;																								\
			else if(tab[i] > (type) saturation)																									\
				tab[i] = (type) saturation + (tab[i] - (type) saturation)*((type)leaking_factor);												\
		}																																		\
		else																																	\
			tab[i] = (type) 0.0f;																												\
	}																																			\
}


#define ReLU_deriv_kernel(name, type)																											\
__global__ void ReLU_deriv_kernel_##name(void *i_deriv, void *i_value, int dim, int biased_dim,	int offset,										\
	 float saturation, float leaking_factor, int length, size_t size)																			\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* deriv = (type*) i_deriv;																												\
	type* value = (type*) i_value;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			if(value[i] <= (type) 0.0f)																											\
				deriv[i] *= leaking_factor;																										\
			else if(value[i] > (type) saturation)																								\
				deriv[i] *= leaking_factor;																										\
		}																																		\
		else																																	\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			if(value[i] <= (type) 0.0f)																											\
				deriv[i] *= leaking_factor;																										\
			else if(value[i] > (type) saturation)																								\
				deriv[i] *= leaking_factor;																										\
		}																																		\
		else																																	\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
}


#define quadratic_deriv_output_error_kernel(name, type)																							\
__global__ void quadratic_deriv_output_error_kernel_##name																						\
	(void *i_delta_o, void *i_output, void *i_target, int dim, int biased_dim, int offset, int length, size_t size, float TC_scale_factor)		\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int nb_filters, c_batch, c_filter, in_filter_pos, pos;																						\
																																				\
	type* delta_o = (type*) i_delta_o;																											\
	type* output  = (type*) i_output;																											\
	type* target  = (type*) i_target;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			pos = i - i/(dim+1);																												\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos]) * TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			nb_filters = size / (dim*offset);																									\
			c_filter = i / (dim*offset);																										\
			c_batch = (i / dim)%offset;																											\
			in_filter_pos = i % dim;																											\
																																				\
			pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;																			\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos]) * TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
}


#define quadratic_output_error_kernel(name, type)																								\
__global__ void quadratic_output_error_kernel_##name																							\
	(float *output_error, void *i_output, void *i_target, int dim, int biased_dim, int offset, int length, size_t size)							\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int nb_filters, c_batch, c_filter, in_filter_pos, pos;																						\
																																				\
	type* output = (type*) i_output;																											\
	type* target = (type*) i_target;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			pos = i - i/(dim+1);																												\
			output_error[i] = (0.5f*((float)output[i] - (float)target[pos])*((float)output[i] - (float)target[pos]));							\
		}																																		\
		else																																	\
			output_error[i]	= 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			nb_filters = size / (dim*offset);																									\
			c_filter = i / (dim*offset);																										\
			c_batch = (i / dim)%offset;																											\
			in_filter_pos = i % dim;																											\
																																				\
			pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;																			\
			output_error[i] = (0.5f*((float)output[i] - (float)target[pos])*((float)output[i] - (float)target[pos]));							\
		}																																		\
		else																																	\
			output_error[i]	= 0.0f;																												\
	}																																			\
}

//#####################################################


//#####################################################
//		  Logistic activation related templates
//#####################################################

#define logistic_activation_kernel(name, type, exp_fct)																							\
__global__ void logistic_activation_kernel_##name(void *i_tab, float beta, float saturation, int dim, 											\
	int biased_dim, int offset, int length, size_t size)																						\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* tab = (type*) i_tab;																													\
	float t_one = (type) 1.0f;																													\
	type t_beta = (type) beta;																													\
	type t_saturation = (type) saturation;																										\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			tab[i] = -t_beta*tab[i];																											\
			if(tab[i] > t_saturation)																											\
				tab[i] = t_saturation;																											\
			tab[i] = t_one/(t_one + exp_fct((float)tab[i]));																					\
		}																																		\
		else																																	\
			tab[i] = (type)0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			tab[i] = -t_beta*tab[i];																											\
			if(tab[i] > t_saturation)																											\
				tab[i] = t_saturation;																											\
			tab[i] = t_one/(t_one + exp_fct((float)tab[i]));																					\
		}																																		\
		else																																	\
			tab[i] = (type)0.0f;																												\
	}																																			\
}


#define logistic_deriv_kernel(name, type)																										\
__global__ void logistic_deriv_kernel_##name(void *i_deriv, void *i_value, float beta, int dim, 												\
	int biased_dim, int offset, int length, size_t size)																						\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
																																				\
	type* deriv = (type*) i_deriv;																												\
	type* value = (type*) i_value;																												\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
			deriv[i] *= (type)beta*value[i]*((type)1.0f-value[i]);																				\
		else																																	\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
			deriv[i] *= (type)beta*value[i]*((type)1.0f-value[i]);																				\
		else																																	\
			deriv[i] = (type) 0.0f;																												\
	}																																			\
}

//#####################################################


//#####################################################
//		  Soft-Max activation related templates
//#####################################################


#define softmax_activation_kernel(name, type, exp_fct)																							\
__global__ void softmax_activation_kernel_##name(void *i_tab, int dim, int biased_dim, 															\
	int offset, int length, int batch_size, size_t size)																						\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int j, k, l;																																\
	int nb_filters;																																\
	type *pos, *off_pos;																														\
	type vmax;																																	\
	float normal = 0.0f;																														\
	type* tab = (type*) i_tab;																													\
																																				\
	if(i >= batch_size)																															\
		return;																																	\
																																				\
	pos = tab + i*(biased_dim);																													\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < length)																															\
		{																																		\
			vmax = *pos;																														\
			for(j = 0; j < dim; j++)																											\
			{																																	\
				off_pos = pos + j;																												\
				if(*off_pos > vmax)																												\
					vmax = *off_pos;																											\
			}																																	\
																																				\
			for(j = 0; j < dim; j++)																											\
			{																																	\
				off_pos = pos + j;																												\
				*off_pos = exp_fct((float)(*off_pos-vmax));																						\
				normal += (float)*off_pos;																										\
			}																																	\
			pos[dim] = 0.0f;																													\
																																				\
			for(j = 0; j < dim; j++)																											\
			{																																	\
				off_pos = pos + j;																												\
				*off_pos = (type)((float)*off_pos/normal);																						\
			}																																	\
			pos[dim] = 0.0f;																													\
		}																																		\
		else																																	\
		{																																		\
			for(j = 0; j < dim; j++)																											\
			{																																	\
				off_pos = pos + j;																												\
				*off_pos = 0.0f;																												\
			}																																	\
			pos[dim] = 0.0f;																													\
		}																																		\
	}																																			\
	else																																		\
	{																																			\
		nb_filters = size / (dim*batch_size);																									\
		if(i < length)																															\
		{																																		\
			vmax = *pos;																														\
			for(k = 0; k < nb_filters ; k++)																									\
			{																																	\
				for(l = 0; l < dim; l++)																										\
				{																																\
					off_pos = pos + k*dim*batch_size + l;																						\
					if(*off_pos > vmax)																											\
						vmax = *off_pos;																										\
				}																																\
			}																																	\
																																				\
			for(k = 0; k < nb_filters ; k++)																									\
			{																																	\
				for(l = 0; l < dim; l++)																										\
				{																																\
					off_pos = pos + k*dim*batch_size + l;																						\
					*off_pos = exp_fct((float)(*off_pos-vmax));																					\
					normal += (float)*off_pos;																									\
				}																																\
			}																																	\
																																				\
			for(k = 0; k < nb_filters ; k++)																									\
			{																																	\
				for(l = 0; l < dim; l++)																										\
				{																																\
					off_pos = pos + k*dim*batch_size + l;																						\
					*off_pos = (type)((float)*off_pos/normal);																					\
				}																																\
			}																																	\
		}																																		\
		else																																	\
		{																																		\
			for(k = 0; k < nb_filters ; k++)																									\
			{																																	\
				for(l = 0; l < dim; l++)																										\
				{																																\
					off_pos = pos + k*dim*batch_size + l;																						\
					*off_pos = 0.0f;																											\
				}																																\
				}																																	\
			}																																		\
		}																																			\
}


#define cross_entropy_deriv_output_error_kernel(name, type)																						\
__global__ void cross_entropy_deriv_output_error_kernel_##name																					\
	(void *i_delta_o, void *i_output, void *i_target, int dim, int biased_dim, int offset, int length, size_t size, float TC_scale_factor)		\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int nb_filters, c_batch, c_filter, in_filter_pos, pos;																						\
																																				\
	type* delta_o = (type*)i_delta_o;																											\
	type* output  = (type*)i_output;																											\
	type* target  = (type*)i_target;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			pos = i - i/(dim+1);																												\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos])* TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			nb_filters = size / (dim*offset);																									\
			c_filter = i / (dim*offset);																										\
			c_batch = (i / dim)%offset;																											\
			in_filter_pos = i % dim;																											\
																																				\
			pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;																			\
			delta_o[i] = (type)(((float)output[i] - (float)target[pos])* TC_scale_factor);														\
		}																																		\
		else																																	\
			delta_o[i] = (type) 0.0f;																											\
	}																																			\
}


#define cross_entropy_output_error_kernel(name, type)																							\
__global__ void cross_entropy_output_error_kernel_##name																						\
	(float *output_error, void *i_output, void *i_target, int dim, int biased_dim, int offset, int length, size_t size)							\
{																																				\
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	int nb_filters, c_batch, c_filter, in_filter_pos, pos;																						\
																																				\
	type* output  = (type*)i_output;																											\
	type* target  = (type*)i_target;																											\
																																				\
	if(i >= size)																																\
		return;																																	\
																																				\
	if(biased_dim > dim)																														\
	{																																			\
		if(i < (length*biased_dim) && (i+1)%(dim+1) != 0)																						\
		{																																		\
			pos = i - i/(dim+1);																												\
			if((float)output[i] > 0.000001f)																									\
				output_error[i] = -(float)target[pos] * logf((float)output[i]);																	\
			else																																\
				output_error[i] = -(float)target[pos] * logf((float)0.000001f);																	\
		}																																		\
		else																																	\
			output_error[i] = 0.0f;																												\
	}																																			\
	else																																		\
	{																																			\
		if((i / dim)%offset < length)																											\
		{																																		\
			nb_filters = size / (dim*offset);																									\
			c_filter = i / (dim*offset);																										\
			c_batch = (i / dim)%offset;																											\
			in_filter_pos = i % dim;																											\
																																				\
			pos = in_filter_pos + (c_filter + c_batch*nb_filters)*dim;																			\
			if((float)output[i] > 0.000001f)																									\
				output_error[i] = -(float)target[pos] * logf((float)output[i]);																	\
			else																																\
				output_error[i] = -(float)target[pos] * logf((float)0.000001f);																	\
		}																																		\
		else																																	\
			output_error[i] = 0.0f;																												\
	}																																			\
}


//#####################################################
//		  YOLO activation related templates
//#####################################################

#define YOLO_activation_kernel(name, type, exp_fct)																								\
__global__ void YOLO_activation_kernel_##name(void *i_tab, int flat_offset, size_t len, yolo_param y_param, size_t size, int class_softmax)		\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i >= size)																																\
		return;																																	\
																																				\
	type *tab = (type*) i_tab;																													\
																																				\
	int nb_class = y_param.nb_class, nb_param = y_param.nb_param, nb_angle = y_param.nb_angle;													\
	/*Default values are in activ_function.c (set_yolo_params)*/																				\
	float **sm_tab = y_param.slopes_and_maxes_tab;																								\
	float normal = 0.0f;																														\
	type vmax;																																	\
	int fit_dim = y_param.fit_dim;																												\
	int col, in_col, j, output_offset;																											\
																																				\
	output_offset = 8+nb_class+nb_param+nb_angle;																								\
	col = i / flat_offset;																														\
	in_col = col%output_offset;																													\
																																				\
	/*Position*/																																\
	if(in_col >= 0 && in_col < 3)																												\
	{																																			\
		if(fit_dim > in_col)																													\
		{																																		\
			tab[i] = -(type)sm_tab[0][0]*tab[i];																								\
			if(tab[i] > (type)sm_tab[0][1])																										\
				tab[i] = (type)sm_tab[0][1];																									\
			else if(tab[i] < (type)sm_tab[0][2])																								\
				tab[i] = (type)sm_tab[0][2];																									\
			tab[i] = 1.0f/(1.0f + exp_fct(tab[i]));																								\
		}																																		\
		else																																	\
			tab[i] = 0.5f; /*Center of the cell*/																								\
		return;																																	\
	}																																			\
																																				\
	/*Box size*/																																\
	if(in_col >= 3 && in_col < 6)																												\
	{																																			\
		if(fit_dim > in_col-3)																													\
		{																																		\
			tab[i] = (type)sm_tab[1][0]*tab[i];																									\
			if(tab[i] > (type)sm_tab[1][1])																										\
				tab[i] = (type)sm_tab[1][1];																									\
			else if(tab[i] < (type)(sm_tab[1][2]))																								\
				tab[i] = (sm_tab[1][2]);																										\
		}																																		\
		else																																	\
			tab[i] = 0.0f; /*Output = prior*/																									\
		return;																																	\
	}																																			\
																																				\
	/*Object probability*/																														\
	if(in_col == 6)																																\
	{																																			\
		tab[i] = -(type)sm_tab[2][0]*tab[i];																									\
		if(tab[i] > (type)sm_tab[2][1])																											\
			tab[i] = (type)sm_tab[2][1];																										\
		else if(tab[i] < (type)sm_tab[2][2])																									\
			tab[i] = (type)sm_tab[2][2];																										\
		tab[i] = 1.0f/(1.0f + exp_fct(tab[i]));																									\
		return;																																	\
	}																																			\
																																				\
	/*Objectness (Obj. quality => based on IoU)*/																								\
	if(in_col == 7)																																\
	{																																			\
		tab[i] = -(type)sm_tab[3][0]*tab[i];																									\
		if(tab[i] > (type)sm_tab[3][1])																											\
			tab[i] = (type)sm_tab[3][1];																										\
		else if(tab[i] < (type)sm_tab[3][2])																									\
			tab[i] = (type)sm_tab[3][2];																										\
		tab[i] = 1.0f/(1.0f + exp_fct(tab[i]));																									\
		return;																																	\
	}																																			\
																																				\
	/*Classes*/																																	\
	if(in_col >= 8 && in_col < 8+nb_class)																										\
	{																																			\
		if(class_softmax)																														\
		{																																		\
			if(in_col != 8)																														\
				return;																															\
			vmax = tab[i];																														\
			for(j = 1; j < nb_class; j++)																										\
				if(tab[i+j*flat_offset] > vmax)																									\
					vmax = tab[i+j*flat_offset];																								\
																																				\
			for(j = 0; j < nb_class; j++)																										\
			{																																	\
				tab[i+j*flat_offset] = exp_fct((tab[i+j*flat_offset]-vmax));																	\
				normal += (float)tab[i+j*flat_offset];																							\
			}																																	\
																																				\
			for(j = 0; j < nb_class; j++)																										\
				tab[i+j*flat_offset] = (type)((float)tab[i+j*flat_offset]/normal);																\
		}																																		\
		else																																	\
		{																																		\
			tab[i] = -(type)sm_tab[4][0]*tab[i];																								\
			if(tab[i] > (type)sm_tab[4][1])																										\
				tab[i] = (type)sm_tab[4][1];																									\
			else if(tab[i] < (type)sm_tab[4][2])																								\
				tab[i] = (type)sm_tab[4][2];																									\
			tab[i] = 1.0f/(1.0f + exp_fct(tab[i]));																								\
		}																																		\
		return;																																	\
	}																																			\
																																				\
	/*Additional parameters (regression)*/																										\
	if(in_col >= 8+nb_class && in_col < 8+nb_class+nb_param)																					\
	{																																			\
		tab[i] = (type)sm_tab[5][0]*tab[i];																										\
		if(tab[i] > (type)sm_tab[5][1])																											\
			tab[i] = (type)sm_tab[5][1];																										\
		else if(tab[i] < (type)(sm_tab[5][2]))																									\
			tab[i] = (sm_tab[5][2]);																											\
		return;																																	\
	}																																			\
	/*Encoded angle head regression*/																											\
	if(in_col >= 8+nb_class+nb_param)																											\
	{																																			\
		tab[i] = (type)y_param.angle_sm[0]*tab[i];																								\
		if(tab[i] > (type)y_param.angle_sm[1])																									\
			tab[i] = (type)y_param.angle_sm[1];																									\
		else if(tab[i] < (type)y_param.angle_sm[2])																								\
			tab[i] = (type)y_param.angle_sm[2];																									\
		return;																																	\
	}																																			\
}

__device__ float gpu_IoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d;
	
	inter_w = max(0.0f, min(output[3], target[3]) - max(output[0], target[0]));
	inter_h = max(0.0f, min(output[4], target[4]) - max(output[1], target[1]));
	inter_d = max(0.0f, min(output[5], target[5]) - max(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d =  fabsf(output[3]-output[0])*fabsf(output[4]-output[1])*fabsf(output[5]-output[2])
			+ fabsf(target[3]-target[0])*fabsf(target[4]-target[1])*fabsf(target[5]-target[2])
			- inter_3d;
	
	return ((float)inter_3d)/(float)uni_3d;
}


__device__ float gpu_GIoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_3d, enclose_w, enclose_h, enclose_d;
	
	inter_w = max(0.0f, min(output[3], target[3]) - max(output[0], target[0]));
	inter_h = max(0.0f, min(output[4], target[4]) - max(output[1], target[1]));
	inter_d = max(0.0f, min(output[5], target[5]) - max(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d =  fabsf(output[3]-output[0])*fabsf(output[4]-output[1])*fabsf(output[5]-output[2])
			+ fabsf(target[3]-target[0])*fabsf(target[4]-target[1])*fabsf(target[5]-target[2])
			- inter_3d;
	enclose_w = (max(output[3], target[3]) - min(output[0], target[0]));
	enclose_h = (max(output[4], target[4]) - min(output[1], target[1]));
	enclose_d = (max(output[5], target[5]) - min(output[2], target[2]));
	enclose_3d = enclose_w * enclose_h * enclose_d;
	
	return (((float)inter_3d)/(float)uni_3d - (float)(enclose_3d - uni_3d)/(float)enclose_3d);
}

//order: xmin, ymin, zmin, xmax, ymax, zmax
// Take into acount the distance in IoU, useful for crowded images
// or to put the emhasis on positionning in objectness score
__device__ float gpu_DIoU_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_w, enclose_h, enclose_d;
	float cx_a, cx_b, cy_a, cy_b, cz_a, cz_b, dist_cent, diag_enclose;
	
	inter_w = max(0.0f, min(output[3], target[3]) - max(output[0], target[0]));
	inter_h = max(0.0f, min(output[4], target[4]) - max(output[1], target[1]));
	inter_d = max(0.0f, min(output[5], target[5]) - max(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d =  fabsf(output[3]-output[0])*fabsf(output[4]-output[1])*fabsf(output[5]-output[2])
			+ fabsf(target[3]-target[0])*fabsf(target[4]-target[1])*fabsf(target[5]-target[2])
			- inter_3d;
	enclose_w = (max(output[3], target[3]) - min(output[0], target[0]));
	enclose_h = (max(output[4], target[4]) - min(output[1], target[1]));
	enclose_d = (max(output[5], target[5]) - min(output[2], target[2]));
	
	cx_a = (output[3] + output[0])*0.5; cx_b = (target[3] + target[0])*0.5; 
	cy_a = (output[4] + output[1])*0.5; cy_b = (target[4] + target[1])*0.5;
	cz_a = (output[5] + output[2])*0.5; cz_b = (target[5] + target[2])*0.5;
	dist_cent = sqrt((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b) + (cz_a - cz_b)*(cz_a - cz_b));
	diag_enclose = sqrt(enclose_w*enclose_w + enclose_h*enclose_h + enclose_d*enclose_d);
	
	return ((float)inter_3d)/(float)uni_3d - (float)(dist_cent/diag_enclose);
}

// Distance penalty is less in this version for a given distance between boxes
// More suited for usual VOC images, or sparse astro images
__device__ float gpu_DIoU2_fct(float *output, float *target)
{
	float inter_w, inter_h, inter_d, inter_3d, uni_3d, enclose_w, enclose_h, enclose_d;
	float cx_a, cx_b, cy_a, cy_b, cz_a, cz_b, dist_cent, diag_enclose;
	
	inter_w = max(0.0f, min(output[3], target[3]) - max(output[0], target[0]));
	inter_h = max(0.0f, min(output[4], target[4]) - max(output[1], target[1]));
	inter_d = max(0.0f, min(output[5], target[5]) - max(output[2], target[2]));
	
	inter_3d = inter_w * inter_h * inter_d;
	uni_3d =  fabsf(output[3]-output[0])*fabsf(output[4]-output[1])*fabsf(output[5]-output[2])
			+ fabsf(target[3]-target[0])*fabsf(target[4]-target[1])*fabsf(target[5]-target[2])
			- inter_3d;
	enclose_w = (max(output[3], target[3]) - min(output[0], target[0]));
	enclose_h = (max(output[4], target[4]) - min(output[1], target[1]));
	enclose_d = (max(output[5], target[5]) - min(output[2], target[2]));
	
	cx_a = (output[3] + output[0])*0.5; cx_b = (target[3] + target[0])*0.5; 
	cy_a = (output[4] + output[1])*0.5; cy_b = (target[4] + target[1])*0.5;
	cz_a = (output[5] + output[2])*0.5; cz_b = (target[5] + target[2])*0.5;
	dist_cent = ((cx_a - cx_b)*(cx_a - cx_b) + (cy_a - cy_b)*(cy_a - cy_b) + (cz_a - cz_b)*(cz_a - cz_b));
	diag_enclose = (enclose_w*enclose_w + enclose_h*enclose_h + enclose_d*enclose_d);
	
	return ((float)inter_3d)/(float)uni_3d - (float)(dist_cent/diag_enclose);
}

__device__ float gpu_yolo_polygon_signed_area(const float *poly, int n)
{
	float area = 0.0f;
	for(int i = 0; i < n; i++)
	{
		int j = (i + 1) % n;
		area += poly[2*i] * poly[2*j+1] - poly[2*i+1] * poly[2*j];
	}
	return 0.5f * area;
}

__device__ float gpu_yolo_polygon_area(const float *poly, int n)
{
	return fabsf(gpu_yolo_polygon_signed_area(poly, n));
}

__device__ int gpu_yolo_inside_half_plane(const float *point, const float *edge_start, const float *edge_end, int clip_ccw)
{
	float edge_x = edge_end[0] - edge_start[0];
	float edge_y = edge_end[1] - edge_start[1];
	float rel_x = point[0] - edge_start[0];
	float rel_y = point[1] - edge_start[1];
	float cross = edge_x * rel_y - edge_y * rel_x;
	return clip_ccw ? (cross >= -1.0e-6f) : (cross <= 1.0e-6f);
}

__device__ void gpu_yolo_line_intersection(const float *p1, const float *p2, const float *q1, const float *q2, float *out)
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

__device__ int gpu_yolo_polygon_clip(const float *subject, int subject_n, const float *clip, int clip_n, float *result)
{
	float input[32], output[32], inter[2];
	int output_n = subject_n;
	int clip_ccw = gpu_yolo_polygon_signed_area(clip, clip_n) >= 0.0f;
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
		int prev_inside = gpu_yolo_inside_half_plane(prev, edge_start, edge_end, clip_ccw);

		for(int i = 0; i < input_n; i++)
		{
			float curr[2] = {input[2*i], input[2*i+1]};
			int curr_inside = gpu_yolo_inside_half_plane(curr, edge_start, edge_end, clip_ccw);
			if(curr_inside)
			{
				if(!prev_inside && output_n < 16)
				{
					gpu_yolo_line_intersection(prev, curr, edge_start, edge_end, inter);
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
				gpu_yolo_line_intersection(prev, curr, edge_start, edge_end, inter);
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

__device__ void gpu_yolo_obb_corners(float cx, float cy, float w, float h, float theta, float *poly)
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

__device__ float gpu_RotatedIoU_fct(float *output, float *target, float output_theta, float target_theta)
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

	gpu_yolo_obb_corners(output_cx, output_cy, output_w, output_h, output_theta, out_poly);
	gpu_yolo_obb_corners(target_cx, target_cy, target_w, target_h, target_theta, targ_poly);
	int inter_n = gpu_yolo_polygon_clip(out_poly, 4, targ_poly, 4, inter_poly);
	float inter_area = (inter_n <= 0) ? 0.0f : gpu_yolo_polygon_area(inter_poly, inter_n);
	float output_area = fmaxf(output_w * output_h, 1.0e-6f);
	float target_area = fmaxf(target_w * target_h, 1.0e-6f);
	float uni = output_area + target_area - inter_area;
	if(uni <= 1.0e-6f)
		return 0.0f;
	return fminf(1.0f, fmaxf(0.0f, inter_area / uni));
}

__device__ void gpu_yolo_load_target_box(float *target, int l_t, float *targ_int, int target_box_mode)
{
	if(target_box_mode > 0)
	{
		for(int l = 0; l < 3; l++)
		{
			float c = (float)target[l_t+1+l];
			float s = fmaxf((float)target[l_t+4+l], 1.0e-6f);
			targ_int[l] = c - 0.5f*s;
			targ_int[l+3] = c + 0.5f*s;
		}
	}
	else
	{
		for(int l = 0; l < 6; l++)
			targ_int[l] = (float)target[l_t+1+l];
	}
}

__device__ float gpu_yolo_obb_cov_loss_terms(
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

#define GPU_YOLO_LOAD_TARGET_BOX(target, l_t, targ_int, target_box_mode)																			\
	do {																																			\
		if((target_box_mode) > 0)																													\
		{																																			\
			for(int _tl = 0; _tl < 3; _tl++)																										\
			{																																		\
				float _tc = (float)(target)[(l_t)+1+_tl];																							\
				float _ts = fmaxf((float)(target)[(l_t)+4+_tl], 1.0e-6f);																			\
				(targ_int)[_tl] = _tc - 0.5f*_ts;																									\
				(targ_int)[_tl+3] = _tc + 0.5f*_ts;																								\
				}																																		\
				}																																			\
		else																																		\
		{																																			\
			for(int _tl = 0; _tl < 6; _tl++)																										\
				(targ_int)[_tl] = (float)(target)[(l_t)+1+_tl];																					\
		}																																			\
	} while(0)

#define GPU_YOLO_BOX_QUALITY(out_int, targ_int, output, target, l_o, l_t, f_offset, nb_class, nb_param, nb_angle, y_param) \
	(((y_param).IoU_type == ROTIOU && (nb_angle) >= 2) ? \
		gpu_RotatedIoU_fct((out_int), (targ_int), \
			0.5f*atan2f((float)(output)[((l_o)+8+(nb_class)+(nb_param)+1)*(f_offset)], (float)(output)[((l_o)+8+(nb_class)+(nb_param)+0)*(f_offset)]), \
			0.5f*atan2f((float)(target)[(l_t)+7+(nb_param)+1], (float)(target)[(l_t)+7+(nb_param)+0])) : \
		(y_param).c_IoU_fct((out_int), (targ_int)))

template<typename type>
__device__ float gpu_yolo_clamp_quality(type value, float floor)
{
	float v = (float)value;
	if(v < floor)
		return floor;
	if(v > 0.98f)
		return 0.98f;
	return v;
}

template<typename type>
__device__ float gpu_yolo_probability_quality_target(type *output, type *target,
	int l_o, int l_t, size_t f_offset, int nb_class, int nb_param, int nb_angle, yolo_param y_param, float max_IoU, float *obj_in_offset)
{
	float floor = y_param.prob_quality_floor;
	float q_box = gpu_yolo_clamp_quality((1.0f + max_IoU) * 0.5f, floor);
	float err_sum = 0.0f, weight_sum = 0.0f;
	float center_err2 = 0.0f;
	int k, dim_count = y_param.fit_dim;

	if(y_param.prob_quality_mode <= 0)
		return 0.98f;
	if(y_param.prob_quality_mode == 1)
		return q_box;
	if(y_param.prob_quality_mode == 3)
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
		return gpu_yolo_clamp_quality(1.0f - sqrtf(center_err2 / (float)dim_count), floor);
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
	return gpu_yolo_clamp_quality(q_box * expf(-y_param.prob_quality_scale * err_sum / weight_sum), floor);
}

template<typename type>
__device__ int gpu_yolo_target_is_da_like(type *target, int l_t, yolo_param y_param)
{
	float flux = (float)target[l_t+7+0];
	float w = fabsf((float)target[l_t+4]);
	float h = fabsf((float)target[l_t+5]);
	float aspect;

	if(y_param.obj_quality_da_flux_max >= 0.0f && flux > y_param.obj_quality_da_flux_max)
		return 0;

	if(y_param.obj_quality_da_bmaj_log_max > y_param.obj_quality_da_bmaj_log_min &&
		y_param.obj_quality_da_bmin_log_max > y_param.obj_quality_da_bmin_log_min)
	{
		float bmaj_log = (float)target[l_t+7+1] *
			(y_param.obj_quality_da_bmaj_log_max - y_param.obj_quality_da_bmaj_log_min) +
			y_param.obj_quality_da_bmaj_log_min;
		float bmin_log = (float)target[l_t+7+2] *
			(y_param.obj_quality_da_bmin_log_max - y_param.obj_quality_da_bmin_log_min) +
			y_param.obj_quality_da_bmin_log_min;
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
	if(y_param.obj_quality_da_aspect_min > 0.0f && aspect < y_param.obj_quality_da_aspect_min)
		return 0;
	if(y_param.obj_quality_da_aspect_max > 0.0f && aspect > y_param.obj_quality_da_aspect_max)
		return 0;
	return 1;
}

template<typename type>
__device__ float gpu_yolo_objectness_quality_target(type *output, type *target,
	int l_o, int l_t, size_t f_offset, int nb_class, int nb_param, yolo_param y_param, float max_IoU, float *obj_in_offset)
{
	float floor = y_param.obj_quality_floor;
	float q_geom = gpu_yolo_clamp_quality((1.0f + max_IoU) * 0.5f, floor);
	float q_center = q_geom;
	float q_phys = q_geom;
	float center_err2 = 0.0f, phys_err = 0.0f;
	float center_weight, geom_weight, phys_weight, weight_sum, quality;
	int k, dim_count = y_param.fit_dim;

	if(y_param.obj_quality_mode <= 0)
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
		q_center = gpu_yolo_clamp_quality(q_center, floor);
	}

	if(nb_param > 0)
	{
		for(k = 0; k < nb_param; k++)
			phys_err += fabsf((float)output[(l_o+8+nb_class+k)*f_offset] - (float)target[l_t+7+k]);
		q_phys = expf(-y_param.obj_quality_scale * phys_err / (float)nb_param);
		q_phys = gpu_yolo_clamp_quality(q_phys, floor);
	}

	center_weight = y_param.obj_quality_center_weight;
	geom_weight = y_param.obj_quality_geom_weight;
	phys_weight = y_param.obj_quality_phys_weight;
	if(y_param.obj_quality_mode == 2 && gpu_yolo_target_is_da_like(target, l_t, y_param))
	{
		center_weight = y_param.obj_quality_da_center_weight;
		geom_weight = y_param.obj_quality_da_geom_weight;
		phys_weight = y_param.obj_quality_da_phys_weight;
	}

	weight_sum = center_weight + geom_weight + phys_weight;
	if(weight_sum <= 1.0e-6f)
		return q_geom;
	quality = (
		center_weight * q_center
		+ geom_weight * q_geom
		+ phys_weight * q_phys
	) / weight_sum;
	return gpu_yolo_clamp_quality(quality, floor);
}

__device__ float gpu_yolo_smooth_l1_grad(float x)
{
	if(x > 1.0f)
		return 1.0f;
	if(x < -1.0f)
		return -1.0f;
	return x;
}

template<typename type>
__device__ void gpu_yolo_add_scorer_aux_delta(type *delta_o, type *output, type *target,
	int l_o, int l_t, size_t f_offset, int nb_class, int nb_param, yolo_param y_param,
	float *obj_in_offset, int *cell_size, float **sm_tab, float coord_scale, float param_scale,
	float *param_ind_scale, float TC_scale_factor, float max_IoU, float min_param_IoU_lim,
	int diff_flag, int targ_diff_flag)
{
	float sample_weight, bmaj_log_t, bmin_log_t, bmaj_t, bmin_t, conv_arcsec, conv_pix;
	float flux_range, bmaj_range, bmin_range;
	float err, grad, outv, scale;
	int k, dim_count;

	if(y_param.scorer_aux_mode <= 0 || nb_param < 3)
		return;
	if(max_IoU <= min_param_IoU_lim || (diff_flag != 0 && targ_diff_flag >= 2))
		return;

	sample_weight = gpu_yolo_target_is_da_like(target, l_t, y_param) ? y_param.scorer_aux_da_weight : 1.0f;
	if(sample_weight <= 0.0f)
		sample_weight = 1.0f;

	bmaj_range = y_param.obj_quality_da_bmaj_log_max - y_param.obj_quality_da_bmaj_log_min;
	bmin_range = y_param.obj_quality_da_bmin_log_max - y_param.obj_quality_da_bmin_log_min;
	conv_pix = 8.0f;
	if(bmaj_range > 1.0e-6f && bmin_range > 1.0e-6f && y_param.scorer_aux_pixel_arcsec > 0.0f)
	{
		bmaj_log_t = (float)target[l_t+7+1] * bmaj_range + y_param.obj_quality_da_bmaj_log_min;
		bmin_log_t = (float)target[l_t+7+2] * bmin_range + y_param.obj_quality_da_bmin_log_min;
		bmaj_t = expf(bmaj_log_t);
		bmin_t = expf(bmin_log_t);
		conv_arcsec = sqrtf(fmaxf(bmaj_t, bmin_t) * fmaxf(bmaj_t, bmin_t)
			+ y_param.scorer_aux_beam_arcsec * y_param.scorer_aux_beam_arcsec);
		conv_pix = fmaxf(1.0f, conv_arcsec / y_param.scorer_aux_pixel_arcsec);
	}

	if(y_param.scorer_aux_center_scale > 0.0f)
	{
		dim_count = y_param.fit_dim;
		if(dim_count > 2)
			dim_count = 2;
		for(k = 0; k < dim_count; k++)
		{
			outv = (float)output[(l_o+k)*f_offset];
			scale = ((float)cell_size[k]) / conv_pix;
			err = (outv - obj_in_offset[k]) * scale;
			grad = gpu_yolo_smooth_l1_grad(err) * scale;
			delta_o[(l_o+k)*f_offset] = (type)((float)delta_o[(l_o+k)*f_offset]
				+ sample_weight * y_param.scorer_aux_center_scale * TC_scale_factor
				* sm_tab[0][0] * coord_scale * outv * (1.0f - outv) * grad);
		}
	}

	if(y_param.scorer_aux_flux_scale > 0.0f)
	{
		flux_range = y_param.scorer_aux_flux_log_max - y_param.scorer_aux_flux_log_min;
		if(flux_range <= 1.0e-6f)
			flux_range = 1.0f;
		err = ((float)output[(l_o+8+nb_class+0)*f_offset] - (float)target[l_t+7+0]) * flux_range / 0.50f;
		grad = gpu_yolo_smooth_l1_grad(err) * flux_range / 0.50f;
		delta_o[(l_o+8+nb_class+0)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+0)*f_offset]
			+ sample_weight * y_param.scorer_aux_flux_scale * TC_scale_factor
			* param_ind_scale[0] * sm_tab[5][0] * param_scale * grad);
	}

	if(y_param.scorer_aux_size_scale > 0.0f && bmaj_range > 1.0e-6f && bmin_range > 1.0e-6f)
	{
		err = (((float)output[(l_o+8+nb_class+1)*f_offset] - (float)target[l_t+7+1]) * bmaj_range) / 0.35f;
		grad = gpu_yolo_smooth_l1_grad(err) * bmaj_range / 0.35f;
		delta_o[(l_o+8+nb_class+1)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+1)*f_offset]
			+ sample_weight * y_param.scorer_aux_size_scale * TC_scale_factor
			* param_ind_scale[1] * sm_tab[5][0] * param_scale * grad);

		err = (((float)output[(l_o+8+nb_class+2)*f_offset] - (float)target[l_t+7+2]) * bmin_range) / 0.35f;
		grad = gpu_yolo_smooth_l1_grad(err) * bmin_range / 0.35f;
		delta_o[(l_o+8+nb_class+2)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+2)*f_offset]
			+ sample_weight * y_param.scorer_aux_size_scale * TC_scale_factor
			* param_ind_scale[2] * sm_tab[5][0] * param_scale * grad);
	}
}

__device__ int gpu_yolo_flux_refine_is_delta_param(yolo_param y_param, int k, int nb_param)
{
	return (
		y_param.flux_refine_mode > 0
		&& nb_param >= 4
		&& y_param.flux_refine_delta_param_index >= 0
		&& y_param.flux_refine_delta_param_index < nb_param
		&& k == y_param.flux_refine_delta_param_index
	);
}

__device__ int gpu_yolo_flux_refine_is_gate_param(yolo_param y_param, int k, int nb_param)
{
	return (
		y_param.flux_refine_mode >= 2
		&& nb_param >= 5
		&& y_param.flux_refine_gate_param_index >= 0
		&& y_param.flux_refine_gate_param_index < nb_param
		&& k == y_param.flux_refine_gate_param_index
	);
}

__device__ int gpu_yolo_flux_refine_is_aux_param(yolo_param y_param, int k, int nb_param)
{
	return (
		gpu_yolo_flux_refine_is_delta_param(y_param, k, nb_param)
		|| gpu_yolo_flux_refine_is_gate_param(y_param, k, nb_param)
	);
}

template<typename type>
__device__ void gpu_yolo_add_flux_refine_delta(type *delta_o, type *output, type *target,
	int l_o, int l_t, size_t f_offset, int nb_class, int nb_param, yolo_param y_param,
	float **sm_tab, float param_scale, float *param_ind_scale, float TC_scale_factor,
	float max_IoU, float min_param_IoU_lim, int diff_flag, int targ_diff_flag)
{
	int delta_k = y_param.flux_refine_delta_param_index;
	int gate_k = y_param.flux_refine_gate_param_index;
	float flux_range, base_norm, delta_raw, gate_raw, gate, final_norm, target_norm;
	float residual_norm, delta_norm, ungated_abs_err, base_abs_err, gate_target;
	float err, grad, delta_scale, gate_margin, d_gate;

	if(y_param.flux_refine_mode <= 0 || nb_param < 4)
		return;
	if(delta_k < 0 || delta_k >= nb_param)
		return;
	if(max_IoU <= min_param_IoU_lim || (diff_flag != 0 && targ_diff_flag >= 2))
		return;

	delta_scale = (y_param.flux_refine_delta_norm_scale > 0.0f) ? y_param.flux_refine_delta_norm_scale : 0.25f;
	flux_range = y_param.scorer_aux_flux_log_max - y_param.scorer_aux_flux_log_min;
	if(flux_range <= 1.0e-6f)
		flux_range = 1.0f;

	base_norm = (float)output[(l_o+8+nb_class+0)*f_offset];
	delta_raw = (float)output[(l_o+8+nb_class+delta_k)*f_offset];
	target_norm = (float)target[l_t+7+0];

	if(y_param.flux_refine_mode < 2)
	{
		if(y_param.flux_refine_loss_scale <= 0.0f)
			return;
		final_norm = base_norm + delta_scale * delta_raw;
		err = (final_norm - target_norm) * flux_range / 0.50f;
		grad = gpu_yolo_smooth_l1_grad(err) * flux_range / 0.50f * delta_scale;
		delta_o[(l_o+8+nb_class+delta_k)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+delta_k)*f_offset]
			+ y_param.flux_refine_loss_scale * TC_scale_factor
			* param_ind_scale[delta_k] * sm_tab[5][0] * param_scale * grad);

		if(y_param.flux_refine_detach_base == 0)
		{
			grad = gpu_yolo_smooth_l1_grad(err) * flux_range / 0.50f;
			delta_o[(l_o+8+nb_class+0)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+0)*f_offset]
				+ y_param.flux_refine_loss_scale * TC_scale_factor
				* param_ind_scale[0] * sm_tab[5][0] * param_scale * grad);
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

	if(y_param.flux_refine_loss_scale > 0.0f)
	{
		err = (delta_norm - residual_norm) * flux_range / 0.50f;
		grad = gpu_yolo_smooth_l1_grad(err) * flux_range / 0.50f * delta_scale;
		delta_o[(l_o+8+nb_class+delta_k)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+delta_k)*f_offset]
			+ y_param.flux_refine_loss_scale * TC_scale_factor
			* param_ind_scale[delta_k] * sm_tab[5][0] * param_scale * grad);
	}

	if(y_param.flux_refine_gate_loss_scale > 0.0f)
	{
		gate_margin = (y_param.flux_refine_gate_margin_norm > 0.0f) ? y_param.flux_refine_gate_margin_norm : 0.01f;
		base_abs_err = fabsf(residual_norm);
		ungated_abs_err = fabsf(delta_norm - residual_norm);
		gate_target = (ungated_abs_err + gate_margin < base_abs_err) ? 1.0f : 0.0f;
		grad = gate_raw - gate_target;
		delta_o[(l_o+8+nb_class+gate_k)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+gate_k)*f_offset]
			+ y_param.flux_refine_gate_loss_scale * TC_scale_factor
			* param_ind_scale[gate_k] * sm_tab[5][0] * param_scale * grad);
	}

	if(y_param.flux_refine_final_loss_scale > 0.0f)
	{
		final_norm = base_norm + gate * delta_norm;
		err = (final_norm - target_norm) * flux_range / 0.50f;
		grad = gpu_yolo_smooth_l1_grad(err) * flux_range / 0.50f;
		delta_o[(l_o+8+nb_class+delta_k)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+delta_k)*f_offset]
			+ y_param.flux_refine_final_loss_scale * TC_scale_factor
			* param_ind_scale[delta_k] * sm_tab[5][0] * param_scale * grad * gate * delta_scale);
		delta_o[(l_o+8+nb_class+gate_k)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+gate_k)*f_offset]
			+ y_param.flux_refine_final_loss_scale * TC_scale_factor
			* param_ind_scale[gate_k] * sm_tab[5][0] * param_scale * grad * delta_norm * d_gate);
		if(y_param.flux_refine_detach_base == 0)
		{
			delta_o[(l_o+8+nb_class+0)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+0)*f_offset]
				+ y_param.flux_refine_final_loss_scale * TC_scale_factor
				* param_ind_scale[0] * sm_tab[5][0] * param_scale * grad);
		}
	}
}


typedef float(*pointFunction_gpu_IoU)(float*, float*); 
__device__ pointFunction_gpu_IoU device_gpu_IoU_fct  = gpu_IoU_fct;
__device__ pointFunction_gpu_IoU device_gpu_GIoU_fct = gpu_GIoU_fct;
__device__ pointFunction_gpu_IoU device_gpu_DIoU_fct = gpu_DIoU_fct;
__device__ pointFunction_gpu_IoU device_gpu_DIoU2_fct = gpu_DIoU2_fct;


#define YOLO_deriv_error_kernel(name, type)																										\
__global__ void YOLO_deriv_error_kernel_##name																									\
	(void *i_delta_o, void *i_output, void *i_target, int flat_target_size, int flat_output_size, 												\
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, size_t size, float TC_scale_factor, int nb_im_iter)						\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i >= size)																																\
		return;																																	\
																																				\
	type *delta_o = (type*) i_delta_o;																											\
	type *output  = (type*) i_output;																											\
	type *target  = (type*) i_target;																											\
																																				\
	/* Define many "shorts" for y_param content to enhance code redeability*/																	\
	int nb_box                      = y_param.nb_box;																							\
	int nb_class                    = y_param.nb_class;																							\
	int nb_param                    = y_param.nb_param; 																						\
	int nb_angle                    = y_param.nb_angle; 																						\
	int strict_box_size_association = y_param.strict_box_size_association;																		\
	int fit_dim                     = y_param.fit_dim;																							\
	int rand_startup                = y_param.rand_startup;																						\
	float rand_prob_best_box_assoc  = y_param.rand_prob_best_box_assoc;																			\
	float rand_prob                 = y_param.rand_prob;																						\
	float min_prior_forced_scaling  = y_param.min_prior_forced_scaling;																			\
		int class_softmax               = y_param.class_softmax;																					\
		int diff_flag                   = y_param.diff_flag;																						\
			int prior_dist_type             = y_param.prior_dist_type;																					\
			int target_box_mode             = y_param.target_box_mode;																					\
			int obb_loss_mode               = y_param.obb_loss_mode;																					\
			int multi_pos_topk              = y_param.multi_pos_topk;																					\
																																					\
		float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];																\
		float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];																\
		float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];																\
			float angle_scale = y_param.angle_scale, angle_unit_norm_scale = y_param.angle_unit_norm_scale;												\
			float obb_loss_scale = y_param.obb_loss_scale;																								\
			float multi_pos_iou_ratio = y_param.multi_pos_iou_ratio;																					\
			float multi_pos_min_iou = y_param.multi_pos_min_iou;																						\
			float multi_pos_obj_weight = y_param.multi_pos_obj_weight;																				\
																																				\
	float *prior_size         = y_param.prior_size;																								\
	int   *cell_size          = y_param.cell_size;																								\
	float *param_ind_scale    = y_param.param_ind_scale;																						\
	float *lambda_noobj_prior = y_param.noobj_prob_prior;																						\
	float **sm_tab            = y_param.slopes_and_maxes_tab;																					\
	int   *target_cell_mask   = y_param.target_cell_mask;																						\
	float *IoU_table          = y_param.IoU_table;																								\
	float *dist_prior         = y_param.dist_prior;																								\
	int   *box_locked         = y_param.box_locked;																								\
	float *box_in_pix         = y_param.box_in_pix;																								\
	void *block_state 		  = y_param.block_state;																							\
																																				\
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);																	\
	float good_IoU_lim      = y_param.IoU_limits[0], low_IoU_best_box_assoc = y_param.IoU_limits[1];											\
	float min_prob_IoU_lim  = y_param.IoU_limits[2], min_obj_IoU_lim        = y_param.IoU_limits[3];											\
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim      = y_param.IoU_limits[5];											\
	float min_angle_IoU_lim = y_param.min_angle_IoU_lim;																						\
	float diff_IoU_lim      = y_param.IoU_limits[6], diff_obj_lim           = y_param.IoU_limits[7];											\
	int fit_pos = y_param.fit_parts[0], fit_size  = y_param.fit_parts[1], fit_prob  = y_param.fit_parts[2]; 									\
	int fit_obj = y_param.fit_parts[3], fit_class = y_param.fit_parts[4], fit_param = y_param.fit_parts[5];										\
		int fit_angle = y_param.fit_angle;																											\
		int angle_loss_mode = y_param.angle_loss_mode;																								\
																																				\
	int j, k, l, l_o, l_t;																														\
	size_t f_offset, c_total_nb_area, c_total_nb_area_batch, total_cell_pos_nb_area, total_area_and_cell_offset;								\
		int c_batch, output_offset, target_offset, nb_obj_target, s_p_i = 0;																		\
		int nb_in_cell, id_in_cell, id_in_cell_offset, l_r_b = -1, resp_box = -1, resp_targ = -1, resp_targ_offset, targ_diff_flag = 0;				\
		int secondary_iter, secondary_box;																											\
		float best_dist, c_dist, max_IoU, current_IoU, prob_target, obj_target, multi_pos_iou_thr, secondary_iou, best_secondary_iou;				\
	int cell_pos[3], c_nb_area[3], obj_c[3];																									\
	float *c_box_in_pix, *c_prior_size;																											\
		float obj_in_offset[6], out_int[6], targ_int[6], targ_size[3];																				\
		float obb_grad_logw, obb_grad_logh, obb_grad_theta, obb_theta_norm2;																			\
		float obb_pred_theta, obb_targ_theta, obb_targ_cx, obb_targ_cy;																			\
		float angle_weight, angle_norm, angle_norm_diff, angle_norm_grad;																			\
		float angle_out0, angle_out1, angle_targ0, angle_targ1, angle_dot, angle_common, angle_norm2;												\
	float class_only_IoU = -2.0f;																												\
																																				\
	c_nb_area[0] = nb_area_w; c_nb_area[1] = nb_area_h; c_nb_area[2] = nb_area_d;																\
	c_total_nb_area = c_nb_area[0]*c_nb_area[1]*c_nb_area[2];																					\
	c_batch = i / flat_output_size;																												\
	target += flat_target_size * c_batch;																										\
	f_offset = size;																															\
	output_offset = 8+nb_class+nb_param+nb_angle;																								\
	target_offset = 7+nb_param+((nb_angle > 0) ? nb_angle + 1 : 0)+diff_flag;																	\
																																				\
	i = i % flat_output_size;																													\
	cell_pos[2] = i / (c_nb_area[0]*c_nb_area[1]);																								\
	cell_pos[1] = (int)(i % (c_nb_area[0]*c_nb_area[1])) / c_nb_area[0];																		\
	cell_pos[0] = (int)(i % (c_nb_area[0]*c_nb_area[1])) % c_nb_area[0];																		\
																																				\
	c_total_nb_area_batch = c_total_nb_area * c_batch;																							\
	total_cell_pos_nb_area = cell_pos[2]*c_nb_area[0]*c_nb_area[1] + cell_pos[1]*c_nb_area[0] + cell_pos[0];									\
	total_area_and_cell_offset = c_total_nb_area_batch + total_cell_pos_nb_area;																\
																																				\
	delta_o += total_area_and_cell_offset;																										\
	output  += total_area_and_cell_offset;																										\
																																				\
	target_cell_mask +=	total_area_and_cell_offset * y_param.max_nb_obj_per_image;																\
	/*Could redume memory footprint with a max_nb_obj_per_cell parameter*/																		\
	IoU_table  += total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;															\
	dist_prior += total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;															\
	box_locked += total_area_and_cell_offset * nb_box;																							\
	box_in_pix += total_area_and_cell_offset * 6 * nb_box;																						\
																																				\
	nb_obj_target = target[0];																													\
	target++;																																	\
																																				\
	if(nb_obj_target == -1)																														\
	{																																			\
		nb_obj_target = 1;																														\
		class_only_IoU = good_IoU_lim; 																											\
	}																																			\
																																				\
	best_dist = 1000000000;																														\
	for(k = 0; k < nb_box; k++)																													\
	{																																			\
		box_locked[k] = 0;																														\
		c_box_in_pix = box_in_pix + k*6;																										\
		c_prior_size = prior_size + k*3;																										\
		l_o = k*output_offset;																													\
		for(l = 0; l < 3; l++)																													\
			c_box_in_pix[l] = ((float)output[(l_o+l)*f_offset] + cell_pos[l]) * cell_size[l];													\
		for(l = 0; l < 3; l++)																													\
			c_box_in_pix[l+3] = c_prior_size[l]*expf((float)output[(l_o+l+3)*f_offset]);														\
																																				\
		c_dist = sqrt(c_prior_size[0]*c_prior_size[0] 																							\
					+ c_prior_size[1]*c_prior_size[1]																							\
					+ c_prior_size[2]*c_prior_size[2]);																							\
		if(c_dist < best_dist)																													\
		{																																		\
			best_dist = c_dist;																													\
			s_p_i = k;																															\
		}																																		\
	}																																			\
																																				\
	nb_in_cell = 0;																																\
	for(j = 0; j < nb_obj_target; j++)																											\
	{																																			\
		l_t = j*target_offset;																													\
			GPU_YOLO_LOAD_TARGET_BOX(target, l_t, targ_int, target_box_mode);														\
																																				\
		/* Search for targets that should be predicted by the current cell element */															\
		target_cell_mask[j] = 1;																												\
		for(l = 0; l < 3; l++)																													\
		{																																		\
			obj_c[l] = (int)( (targ_int[l+3] + targ_int[l])*0.5f / cell_size[l]);											\
			/* If target outside the current cell element, set target flag to 0*/																\
			if(obj_c[l] != cell_pos[l])																											\
				target_cell_mask[j] = 0;																										\
		}																																		\
																																				\
		if(target_cell_mask[j] == 1)																											\
			nb_in_cell++;																														\
																																				\
		/* Flag all the "Good but not best boxes" for all targets regardless of the grid element */												\
		for(k = 0; k < nb_box; k++)																												\
		{																																		\
			l_o = k*output_offset;																												\
			if(box_locked[k] != 0)																												\
				continue;																														\
			c_box_in_pix = box_in_pix+k*6;																										\
			for(l = 0; l < 6; l++)																												\
				out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];													\
																																				\
			current_IoU = GPU_YOLO_BOX_QUALITY(out_int, targ_int, output, target, l_o, l_t, f_offset, nb_class, nb_param, nb_angle, y_param);	\
			if(current_IoU > good_IoU_lim)																										\
				box_locked[k] = 1;																												\
		}																																		\
	}																																			\
																																				\
	/* For all targets in cell compute the IoU with the predictions and distances to the priors */												\
	id_in_cell = 0;																																\
	for(j = 0; j < nb_obj_target; j++)																											\
	{																																			\
		id_in_cell_offset = id_in_cell*nb_box;																									\
		if(target_cell_mask[j] == 0)																											\
			continue;																															\
																																				\
		l_t = j*target_offset;																													\
			GPU_YOLO_LOAD_TARGET_BOX(target, l_t, targ_int, target_box_mode);														\
			for(l = 0; l < 3; l++)																													\
				targ_size[l] = targ_int[l+3] - targ_int[l];																							\
																																					\
			for(k = 0; k < nb_box; k++)																												\
			{																																		\
				l_o = k*output_offset;																												\
				c_box_in_pix = box_in_pix+k*6;																										\
				for(l = 0; l < 6; l++)																												\
					out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];													\
																																					\
				current_IoU = GPU_YOLO_BOX_QUALITY(out_int, targ_int, output, target, l_o, l_t, f_offset, nb_class, nb_param, nb_angle, y_param);	\
				IoU_table[id_in_cell_offset + k] = current_IoU;																						\
				dist_prior[id_in_cell_offset + k] = -2.0f;																							\
		}																																		\
																																				\
		/* Restrict the association to the l best theoritical prior (times repetition of identical priors) */									\
		if(strict_box_size_association > 0)																										\
		{																																		\
			if(prior_dist_type == DIST_IOU)																										\
				for(l = 0; l < 6; l++)																											\
					targ_int[l] = copysignf(0.5f,l-2.5f)*targ_size[l%3];																		\
																																				\
			for(k = 0; k < nb_box; k++)																											\
			{																																	\
				c_prior_size = prior_size + k*3;																								\
				switch(prior_dist_type)																											\
				{																																\
					case DIST_IOU:																												\
						for(l = 0; l < 6; l++)																									\
							out_int[l] = copysignf(0.5f,l-2.5f)*c_prior_size[l%3];																\
						dist_prior[id_in_cell_offset + k] = 1.0f - y_param.c_IoU_fct(out_int, targ_int);										\
						break;																													\
																																				\
					default:																													\
					case DIST_SIZE:																												\
						dist_prior[id_in_cell_offset + k] = sqrt(																				\
							 (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])														\
							+(targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])														\
							+(targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]));													\
						break;																													\
																																				\
					case DIST_OFFSET:																											\
						for(l = 0; l < 3; l++)																									\
						{																														\
							obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];																	\
							if(obj_in_offset[l+3] < size_min_sat)																				\
								obj_in_offset[l+3] = logf(size_min_sat);																		\
							else if(obj_in_offset[l+3] > size_max_sat)																			\
								obj_in_offset[l+3] = logf(size_max_sat);																		\
							else																												\
								obj_in_offset[l+3] = logf(obj_in_offset[l+3]);																	\
						}																														\
																																				\
						dist_prior[id_in_cell_offset + k] = 																					\
							 fabsf(obj_in_offset[3])																							\
							+fabsf(obj_in_offset[4])																							\
							+fabsf(obj_in_offset[5]);																							\
						break;																													\
				}																																\
			}																																	\
																																				\
			for(l = 0; l < strict_box_size_association; l++)																					\
			{																																	\
				best_dist = 1000000.0f;																											\
				for(k = 0; k < nb_box; k++)																										\
					if(dist_prior[id_in_cell_offset+k] > 0.0 && dist_prior[id_in_cell_offset+k] < best_dist)									\
						best_dist = dist_prior[id_in_cell_offset+k];																			\
				for(k = 0; k < nb_box; k++) /* Flag the closest theoritical prior (and identical ones if any) */								\
					if(fabsf(dist_prior[id_in_cell_offset+k] - best_dist) < 0.001f )															\
						dist_prior[id_in_cell_offset+k] = -2.0f;																				\
			}																																	\
		}																																		\
																																				\
		id_in_cell++;																															\
	}																																			\
																																				\
	for(id_in_cell = 0; id_in_cell < nb_in_cell; id_in_cell++)																					\
	{																																			\
		/* Force a random box association with only criteria being that the box is not already used */											\
		/* Used as a startup phase to get all the priors closer to the objects to detect */														\
		if(nb_im_iter <= rand_startup)																											\
		{																																		\
			resp_targ = id_in_cell;	resp_box = -1;																								\
			for(k = 0; k < 2*nb_box; k++)																										\
			{																																	\
				resp_box = (int)(curand_uniform(&(((curandState_t*)block_state)[i]))*nb_box);													\
				if(box_locked[resp_box] != 2)																									\
					break;																														\
				resp_box = -1;																													\
			}																																	\
																																				\
			if(resp_box == -1)																													\
				continue;																														\
																																				\
			l = 0;																																\
			for(j = 0; j < nb_obj_target; j++)																									\
			{																																	\
				l += target_cell_mask[j];																										\
				if(l == resp_targ + 1)																											\
					break;																														\
				}																																	\
				l_t = j*target_offset;																												\
				resp_targ_offset = resp_targ*nb_box;																								\
			}																																		\
		else																																	\
		{																																		\
			max_IoU = -2.0f; resp_box = -1;	resp_targ = -1;																						\
			for(j = 0; j < nb_in_cell; j++)																										\
				for(k = 0; k < nb_box; k++)																										\
					if(IoU_table[j*nb_box+k] > max_IoU && dist_prior[j*nb_box+k] < -1.0)														\
					{																															\
						max_IoU = IoU_table[j*nb_box+k];																						\
						resp_targ = j;																											\
						resp_box = k;																											\
					}																															\
																																				\
			/* If strict_box_size > 0 and no more good prior is available, or if there is more targets than boxes */							\
			/* In that case all the remaining target are unable to be associated to */ 															\
			/* any other box and the id_in_cell loop must be stoped */																			\
			if(resp_box == -1)																													\
				continue;																														\
																																				\
			/* l is the "best" index in the "in cell" list */																					\
			/* Need to get back the original target index from the "in cell" index */															\
			l = 0;																																\
			for(j = 0; j < nb_obj_target; j++)																									\
			{																																	\
				l += target_cell_mask[j];																										\
				if(l == resp_targ + 1)																											\
					break;																														\
			}																																	\
			/* The appropriate j value is set after this early stop loop */																		\
			l_t = j*target_offset;																												\
			resp_targ_offset = resp_targ*nb_box;																								\
																																				\
			GPU_YOLO_LOAD_TARGET_BOX(target, l_t, targ_int, target_box_mode);														\
			for(l = 0; l < 3; l++)																												\
				targ_size[l] = targ_int[l+3] - targ_int[l];																						\
																																				\
			if(curand_uniform(&(((curandState_t*)block_state)[i])) < rand_prob)																	\
			{																																	\
				for(k = 0; k < 2*nb_box; k++)																									\
				{																																\
					l_r_b = (int)(curand_uniform(&(((curandState_t*)block_state)[i]))*nb_box);													\
					if(box_locked[l_r_b] != 2)																									\
					{																															\
						resp_box = l_r_b;																										\
						break;																													\
					}																															\
				}																																\
			}																																	\
			/* Force the association to the smallest prior (or identical) if the target is too small */											\
			else if(targ_size[0] < min_prior_forced_scaling*prior_size[s_p_i*3+0]																\
				&& targ_size[1] < min_prior_forced_scaling*prior_size[s_p_i*3+1]																\
				&& targ_size[2] < min_prior_forced_scaling*prior_size[s_p_i*3+2])																\
			{																																	\
				max_IoU = -2.0f; 																												\
				for(k = 0; k < nb_box; k++)																										\
				{																																\
					c_prior_size = prior_size + k*3;																							\
					if((prior_size[s_p_i*3+0] == c_prior_size[0] 																					\
						&& prior_size[s_p_i*3+1] == c_prior_size[1] 																				\
						&& prior_size[s_p_i*3+2] == c_prior_size[2]) 																				\
						&& IoU_table[resp_targ_offset + k] > max_IoU)																			\
					{																															\
						max_IoU = IoU_table[resp_targ_offset + k];																				\
						resp_box = k;																											\
					}																															\
				}																																\
			}																																	\
			/* If prediction is too bad, associate it to the best theoritical prior instead (might found the same box again) */					\
			/* Also force the best theoritical prior association at a small rate */																\
			else if(max_IoU < low_IoU_best_box_assoc || 																						\
				curand_uniform(&(((curandState_t*)block_state)[i])) < rand_prob_best_box_assoc)													\
			{																																	\
				if(prior_dist_type == DIST_IOU)																									\
					for(l = 0; l < 6; l++)																										\
						targ_int[l] = copysignf(0.5f,l-2.5f)*targ_size[l%3];																	\
																																				\
				best_dist = 100000.0f;																											\
				for(k = 0; k < nb_box; k++)																										\
				{																																\
					c_prior_size = prior_size + k*3;																							\
					switch(prior_dist_type)																										\
					{																															\
						case DIST_IOU:																											\
							for(l = 0; l < 6; l++)																								\
								out_int[l] = copysignf(0.5f,l-2.5f)*c_prior_size[l%3];															\
							dist_prior[resp_targ_offset + k] = 1.0f - y_param.c_IoU_fct(out_int, targ_int);										\
							break;																												\
																																				\
						default:																												\
						case DIST_SIZE:																											\
							dist_prior[resp_targ_offset + k] = sqrt(																			\
								 (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])													\
								+(targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])													\
								+(targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]));												\
							break;																												\
																																				\
						case DIST_OFFSET:																										\
							for(l = 0; l < 3; l++)																								\
							{																													\
								obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];																\
								if(obj_in_offset[l+3] < size_min_sat)																			\
									obj_in_offset[l+3] = logf(size_min_sat);																	\
								else if(obj_in_offset[l+3] > size_max_sat)																		\
									obj_in_offset[l+3] = logf(size_max_sat);																	\
								else																											\
									obj_in_offset[l+3] = logf(obj_in_offset[l+3]);																\
							}																													\
																																				\
							dist_prior[resp_targ_offset + k] =																					\
								 fabsf(obj_in_offset[3])																						\
								+fabsf(obj_in_offset[4])																						\
								+fabsf(obj_in_offset[5]);																						\
							break;																												\
					}																															\
					if(dist_prior[resp_targ_offset + k] < best_dist)																			\
						best_dist = dist_prior[resp_targ_offset + k];																			\
				}																																\
				max_IoU = -2.0f;																												\
				for(k = 0; k < nb_box; k++)																										\
				{																																\
					if(fabsf(dist_prior[resp_targ_offset + k] - best_dist) < 0.001f && IoU_table[resp_targ_offset + k] > max_IoU)				\
					{																															\
						max_IoU = IoU_table[resp_targ_offset + k];																				\
						resp_box = k;																											\
					}																															\
				}																																\
				/* If the best prior (or identical) is not available, the resp_box is unchanged */												\
				/* Should always get a resp_box != -1, regarding all previous conditions */														\
			}																																	\
		}																																		\
																																				\
				c_box_in_pix = box_in_pix + resp_box*6;																									\
				for(l = 0; l < 6; l++)																													\
					out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];														\
																																					\
			GPU_YOLO_LOAD_TARGET_BOX(target, l_t, targ_int, target_box_mode);														\
			for(l = 0; l < 3; l++)																													\
				targ_size[l] = targ_int[l+3] - targ_int[l];																							\
																																					\
			l_o = resp_box*output_offset;																											\
			max_IoU = GPU_YOLO_BOX_QUALITY(out_int, targ_int, output, target, l_o, l_t, f_offset, nb_class, nb_param, nb_angle, y_param);			\
			if(max_IoU > 0.98f)																														\
				max_IoU = 0.98f;																													\
			if(class_only_IoU > -2.0f)																												\
				max_IoU = class_only_IoU; /*regardless of actual IoU because class only box is not precise*/										\
																																					\
			c_prior_size = prior_size + 3*resp_box;																									\
																																				\
		/* Positive reinforcement */ 																											\
		targ_diff_flag = 0;																														\
		if(diff_flag)	/* Cast from mixed precision type to float is always possible, but not necessary to int directly */						\
			targ_diff_flag = (int)((float)target[l_t+7+nb_param+((nb_angle > 0) ? nb_angle + 1 : 0)]);											\
																																				\
			/* If the target is flagged as "difficult", only update the matching box if the prediction is already confident enough */				\
			/* The target is removed from the list anyway, and the corresponding box fall to "background" or "Good_but_not_best" case*/				\
			if(diff_flag && targ_diff_flag > 0 && (max_IoU < diff_IoU_lim || (float)output[(l_o+7)*f_offset] < diff_obj_lim))						\
			{																																		\
				for(k = 0; k < nb_box; k++)																											\
					IoU_table[resp_targ_offset + k] = -2.0f;																						\
				continue;																															\
			}																																		\
																																				\
		/* Mark the box as already associated by removing its contributions to the IoU table */													\
		for(j = 0; j < nb_in_cell; j++)																											\
			IoU_table[j*nb_box + resp_box] = -2.0f;																								\
																																				\
		box_locked[resp_box] = 2;																												\
																																				\
		for(l = 0; l < 3; l++)																													\
			obj_in_offset[l] = ((targ_int[l+3] + targ_int[l])*0.5f - cell_pos[l]*cell_size[l])/(float)cell_size[l];								\
		for(l = 0; l < 3; l++)																													\
		{																																		\
			obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];																					\
			if(obj_in_offset[l+3] < size_min_sat)																								\
				obj_in_offset[l+3] = logf(size_min_sat);																						\
			else if(obj_in_offset[l+3] > size_max_sat)																							\
				obj_in_offset[l+3] = logf(size_max_sat);																						\
			else																																\
				obj_in_offset[l+3] = logf(obj_in_offset[l+3]);																					\
		}																																		\
																																				\
		/* Note: most of the following could be replaced by function pointers to avoid so much switch statements */								\
		switch(fit_pos)																															\
		{																																		\
			case 1:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k && class_only_IoU < -1.9f && (diff_flag == 0 || targ_diff_flag < 3))											\
						delta_o[(l_o+k)*f_offset] = (type)(TC_scale_factor*sm_tab[0][0]															\
							*coord_scale*(float)output[(l_o+k)*f_offset]																		\
							*(1.0f-(float)output[(l_o+k)*f_offset])																				\
							*((float)output[(l_o+k)*f_offset]-obj_in_offset[k]));																\
					else																														\
						delta_o[(l_o+k)*f_offset] = (type)(0.0f);																				\
				}																																\
				break;																															\
			case 0:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k)																												\
						delta_o[(l_o+k)*f_offset] = (type)(TC_scale_factor*sm_tab[0][0]															\
							*coord_scale*(float)output[(l_o+k)*f_offset]																		\
							*(1.0f-(float)output[(l_o+k)*f_offset])																				\
							*((float)output[(l_o+k)*f_offset]-0.5f));																			\
					else																														\
						delta_o[(l_o+k)*f_offset] = (type)(0.0f);																				\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < 3; k++)																											\
					delta_o[(l_o+k)*f_offset] = (type)(0.0f);																					\
				break;																															\
		}																																		\
																																				\
		switch(fit_size)																														\
		{																																		\
			case 1:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k && class_only_IoU < -1.9f && (diff_flag == 0 || targ_diff_flag < 3))											\
						delta_o[(l_o+k+3)*f_offset] = (type) (TC_scale_factor*sm_tab[1][0]														\
							*size_scale*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3]));												\
					else																														\
						delta_o[(l_o+k+3)*f_offset] = (type) (0.0f);																			\
				}																																\
				break;																															\
			case 0:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k)																												\
						delta_o[(l_o+k+3)*f_offset] = (type) (TC_scale_factor*sm_tab[1][0]														\
							*size_scale*((float)output[(l_o+k+3)*f_offset]-0.0f));																\
					else																														\
						delta_o[(l_o+k+3)*f_offset] = (type) (0.0f);																			\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < 3; k++)																											\
					delta_o[(l_o+k+3)*f_offset] = (type) (0.0f);																				\
				break;																															\
		}																																		\
																																				\
		switch(fit_prob)																														\
		{																																		\
			case 1:																																\
				if(max_IoU > min_prob_IoU_lim)																									\
				{																																\
					prob_target = gpu_yolo_probability_quality_target(output, target, l_o, l_t, f_offset, nb_class, nb_param, nb_angle, y_param, max_IoU, obj_in_offset);	\
					delta_o[(l_o+6)*f_offset] = (type)(TC_scale_factor*sm_tab[2][0]																\
						*prob_scale*(float)output[(l_o+6)*f_offset]																				\
						*(1.0f-(float)output[(l_o+6)*f_offset])																					\
						*((float)output[(l_o+6)*f_offset]-prob_target));																		\
				}																																\
				else																															\
					delta_o[(l_o+6)*f_offset] = (type)(0.0f);																					\
				break;																															\
			case 0:																																\
				delta_o[(l_o+6)*f_offset] = (type)(TC_scale_factor*sm_tab[2][0]																	\
					*prob_scale*(float)output[(l_o+6)*f_offset]																					\
					*(1.0f-(float)output[(l_o+6)*f_offset])																						\
					*((float)output[(l_o+6)*f_offset]-0.5f));																					\
				break;																															\
			case -1:																															\
				delta_o[(l_o+6)*f_offset] = (type)(0.0f);																						\
				break;																															\
		}																																		\
																																				\
		switch(fit_obj)																															\
		{																																		\
			case 1:																																\
				if(max_IoU > min_obj_IoU_lim)																									\
				{																																\
					obj_target = gpu_yolo_objectness_quality_target(output, target, l_o, l_t, f_offset, nb_class, nb_param, y_param, max_IoU, obj_in_offset);	\
					delta_o[(l_o+7)*f_offset] = (type)(TC_scale_factor*sm_tab[3][0]																\
						*obj_scale*(float)output[(l_o+7)*f_offset]																				\
						*(1.0f-(float)output[(l_o+7)*f_offset])																					\
						*((float)output[(l_o+7)*f_offset]-obj_target));																			\
				}																																\
				else																															\
					delta_o[(l_o+7)*f_offset] = (type)(0.0f);																					\
				break;																															\
			case 0:																																\
				delta_o[(l_o+7)*f_offset] = (type)(TC_scale_factor*sm_tab[3][0]																	\
					*obj_scale*(float)output[(l_o+7)*f_offset]																					\
					*(1.0f-(float)output[(l_o+7)*f_offset])																						\
					*((float)output[(l_o+7)*f_offset]-0.5f));																					\
				break;																															\
			case -1:																															\
				delta_o[(l_o+7)*f_offset] = (type)(0.0f);																						\
				break;																															\
		}																																		\
																																				\
		/* Note : mean square error on classes => could be changed to soft max but difficult to balance */										\
		switch(fit_class)																														\
		{																																		\
			case 1:																																\
				if(max_IoU > min_class_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))														\
				{																																\
					if(class_softmax)																											\
					{																															\
						for(k = 0; k < nb_class; k++)																							\
						{																														\
							if(k == (int) target[l_t]-1)																						\
								delta_o[(l_o+8+k)*f_offset] = (type) (TC_scale_factor															\
									*class_scale*((float)output[(l_o+8+k)*f_offset]-1.0f));														\
							else																												\
								delta_o[(l_o+8+k)*f_offset] = (type) (TC_scale_factor															\
									*class_scale*((float)output[(l_o+8+k)*f_offset]-0.0f));														\
						}																														\
					}																															\
					else																														\
					{																															\
						for(k = 0; k < nb_class; k++)																							\
						{																														\
							if(k == (int) target[l_t]-1)																						\
								delta_o[(l_o+8+k)*f_offset] = (type) (TC_scale_factor*sm_tab[4][0]												\
									*class_scale*(float)output[(l_o+8+k)*f_offset]																\
									*(1.0f-(float)output[(l_o+8+k)*f_offset])																	\
									*((float)output[(l_o+8+k)*f_offset]-0.98f));																\
							else																												\
								delta_o[(l_o+8+k)*f_offset] = (type) (TC_scale_factor*sm_tab[4][0]												\
									*class_scale*(float)output[(l_o+8+k)*f_offset]																\
									*(1.0f-(float)output[(l_o+8+k)*f_offset])																	\
									*((float)output[(l_o+8+k)*f_offset]-0.02f));																\
						}																														\
					}																															\
				}																																\
				else																															\
					for(k = 0; k < nb_class; k++)																								\
						delta_o[(l_o+8+k)*f_offset] = (type) (0.0f);																			\
				break;																															\
			case 0:																																\
				if(class_softmax)																												\
				{																																\
					/* Could compute CE with target = 1/nb_class, but in this case perfect classification error > 0 (still minimum) */			\
					for(k = 0; k < nb_class; k++)																								\
						delta_o[(l_o+8+k)*f_offset] = (type) (0.0f);																			\
				}																																\
				else																															\
				{																																\
					for(k = 0; k < nb_class; k++)																								\
						delta_o[(l_o+8+k)*f_offset] = (type) (TC_scale_factor*sm_tab[4][0]														\
							*class_scale*(float)output[(l_o+8+k)*f_offset]																		\
							*(1.0f-(float)output[(l_o+8+k)*f_offset])																			\
							*((float)output[(l_o+8+k)*f_offset]-0.5f));																			\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < nb_class; k++)																									\
					delta_o[(l_o+8+k)*f_offset] = (type) (0.0f);																				\
				break;																															\
		}																																		\
																																				\
		/* Linear activation of additional parameters */																						\
			switch(fit_param)																														\
			{																																		\
				case 1:																																\
					if(max_IoU > min_param_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))														\
					for(k = 0; k < nb_param; k++)																								\
					{																															\
						if(gpu_yolo_flux_refine_is_aux_param(y_param, k, nb_param))																\
						{																														\
							delta_o[(l_o+8+nb_class+k)*f_offset] = (type) (0.0f);																\
							continue;																											\
						}																														\
						delta_o[(l_o+8+nb_class+k)*f_offset] = 																					\
							(type) (param_ind_scale[k]*TC_scale_factor*sm_tab[5][0]*param_scale													\
							*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k]));												\
					}																															\
				else																															\
					for(k = 0; k < nb_param; k++)																								\
						delta_o[(l_o+8+nb_class+k)*f_offset] = (type) (0.0f);																	\
				break;																															\
			case 0:																																\
				for(k = 0; k < nb_param; k++)																									\
				{																																\
					if(gpu_yolo_flux_refine_is_aux_param(y_param, k, nb_param))																	\
					{																															\
						delta_o[(l_o+8+nb_class+k)*f_offset] = (type) (0.0f);																	\
						continue;																												\
					}																															\
					delta_o[(l_o+8+nb_class+k)*f_offset] = 																						\
						(type) (param_ind_scale[k]*TC_scale_factor*sm_tab[5][0]*param_scale														\
						*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f));																	\
				}																																\
				break;																															\
				case -1:																															\
					for(k = 0; k < nb_param; k++)																									\
						delta_o[(l_o+8+nb_class+k)*f_offset] = (type) (0.0f);																		\
					break;																															\
			}																																		\
																																					\
			gpu_yolo_add_scorer_aux_delta(delta_o, output, target, l_o, l_t, f_offset, nb_class, nb_param, y_param,								\
				obj_in_offset, cell_size, sm_tab, coord_scale, param_scale, param_ind_scale, TC_scale_factor,										\
				max_IoU, min_param_IoU_lim, diff_flag, targ_diff_flag);																			\
			gpu_yolo_add_flux_refine_delta(delta_o, output, target, l_o, l_t, f_offset, nb_class, nb_param, y_param,								\
				sm_tab, param_scale, param_ind_scale, TC_scale_factor, max_IoU, min_param_IoU_lim, diff_flag, targ_diff_flag);					\
																																					\
			/* Encoded angle head: target layout is [cos2theta, sin2theta, angle_weight]. */														\
			switch(fit_angle)																														\
			{																																		\
				case 1:																																\
					if(nb_angle > 0 && max_IoU > min_angle_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))										\
					{																																\
						angle_weight = (float)target[l_t+7+nb_param+nb_angle];																		\
						if(angle_loss_mode == 1)																									\
						{																															\
							for(k = 0; k < nb_angle; k += 2)																						\
							{																														\
								if(k + 1 >= nb_angle)																								\
								{																													\
									delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] =																	\
										(type)(angle_weight*TC_scale_factor*y_param.angle_sm[0]*angle_scale											\
										*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k]));					\
									continue;																										\
								}																													\
								angle_out0 = (float)output[(l_o+8+nb_class+nb_param+k)*f_offset];													\
								angle_out1 = (float)output[(l_o+8+nb_class+nb_param+k+1)*f_offset];												\
								angle_targ0 = (float)target[l_t+7+nb_param+k];																		\
								angle_targ1 = (float)target[l_t+7+nb_param+k+1];																	\
								angle_norm = sqrtf(angle_out0*angle_out0 + angle_out1*angle_out1);													\
								angle_norm = fmaxf(angle_norm, 1.0e-4f);																			\
								angle_norm2 = angle_norm * angle_norm;																				\
								angle_dot = (angle_out0*angle_targ0 + angle_out1*angle_targ1) / angle_norm;										\
								angle_common = angle_weight*TC_scale_factor*y_param.angle_sm[0]*angle_scale;										\
								delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] =																		\
									(type)(angle_common*(angle_dot*angle_out0/angle_norm2 - angle_targ0/angle_norm));								\
								delta_o[(l_o+8+nb_class+nb_param+k+1)*f_offset] =																	\
									(type)(angle_common*(angle_dot*angle_out1/angle_norm2 - angle_targ1/angle_norm));								\
								if(angle_unit_norm_scale > 0.0f)																					\
								{																													\
									angle_norm_diff = angle_norm - 1.0f;																			\
									angle_norm_grad = TC_scale_factor*angle_unit_norm_scale*angle_norm_diff*(angle_out0/angle_norm);				\
									delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] =																	\
										(type)((float)delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] + angle_norm_grad);								\
									angle_norm_grad = TC_scale_factor*angle_unit_norm_scale*angle_norm_diff*(angle_out1/angle_norm);				\
									delta_o[(l_o+8+nb_class+nb_param+k+1)*f_offset] =																\
										(type)((float)delta_o[(l_o+8+nb_class+nb_param+k+1)*f_offset] + angle_norm_grad);							\
								}																													\
							}																														\
						}																															\
						else																														\
						{																															\
							for(k = 0; k < nb_angle; k++)																							\
								delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] =																		\
									(type)(angle_weight*TC_scale_factor*y_param.angle_sm[0]*angle_scale												\
									*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k]));						\
							if(nb_angle == 2 && angle_unit_norm_scale > 0.0f)																		\
							{																														\
								angle_norm = sqrtf(																									\
									(float)output[(l_o+8+nb_class+nb_param+0)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+0)*f_offset]			\
									+(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]);		\
								angle_norm = fmaxf(angle_norm, 1.0e-8f);																			\
								angle_norm_diff = angle_norm - 1.0f;																				\
								for(k = 0; k < nb_angle; k++)																						\
								{																													\
									angle_norm_grad = TC_scale_factor*angle_unit_norm_scale*angle_norm_diff											\
										*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]/angle_norm);											\
									delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] =																	\
										(type)((float)delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] + angle_norm_grad);								\
								}																													\
							}																														\
						}																															\
					}																																\
					else																															\
						for(k = 0; k < nb_angle; k++)																								\
							delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] = (type) (0.0f);															\
					break;																															\
				case 0:																																\
					for(k = 0; k < nb_angle; k++)																									\
						delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] =																				\
							(type)(TC_scale_factor*y_param.angle_sm[0]*angle_scale*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]));			\
					break;																															\
				case -1:																															\
					for(k = 0; k < nb_angle; k++)																									\
						delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] = (type) (0.0f);																\
					break;																															\
				}																																		\
				if(obb_loss_mode > 0 && obb_loss_scale > 0.0f && nb_angle >= 2																		\
					&& max_IoU > min_angle_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2))														\
				{																																	\
					obb_pred_theta = 0.5f*atan2f((float)output[(l_o+8+nb_class+nb_param+1)*f_offset], (float)output[(l_o+8+nb_class+nb_param+0)*f_offset]);\
					obb_targ_theta = 0.5f*atan2f((float)target[l_t+7+nb_param+1], (float)target[l_t+7+nb_param+0]);								\
					obb_targ_cx = 0.5f*(targ_int[0] + targ_int[3]);																				\
					obb_targ_cy = 0.5f*(targ_int[1] + targ_int[4]);																				\
					gpu_yolo_obb_cov_loss_terms(c_box_in_pix[0], c_box_in_pix[1], c_box_in_pix[3], c_box_in_pix[4], obb_pred_theta, obb_targ_cx, obb_targ_cy, targ_size[0], targ_size[1], obb_targ_theta, &obb_grad_logw, &obb_grad_logh, &obb_grad_theta);\
					delta_o[(l_o+0)*f_offset] = (type)((float)delta_o[(l_o+0)*f_offset] + TC_scale_factor*sm_tab[0][0]*coord_scale*obb_loss_scale*(float)output[(l_o+0)*f_offset]*(1.0f-(float)output[(l_o+0)*f_offset])*((float)output[(l_o+0)*f_offset]-obj_in_offset[0]));\
					delta_o[(l_o+1)*f_offset] = (type)((float)delta_o[(l_o+1)*f_offset] + TC_scale_factor*sm_tab[0][0]*coord_scale*obb_loss_scale*(float)output[(l_o+1)*f_offset]*(1.0f-(float)output[(l_o+1)*f_offset])*((float)output[(l_o+1)*f_offset]-obj_in_offset[1]));\
					delta_o[(l_o+3)*f_offset] = (type)((float)delta_o[(l_o+3)*f_offset] + TC_scale_factor*sm_tab[1][0]*size_scale*obb_loss_scale*obb_grad_logw);\
					delta_o[(l_o+4)*f_offset] = (type)((float)delta_o[(l_o+4)*f_offset] + TC_scale_factor*sm_tab[1][0]*size_scale*obb_loss_scale*obb_grad_logh);\
					obb_theta_norm2 = fmaxf((float)output[(l_o+8+nb_class+nb_param+0)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+0)*f_offset] +(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+1)*f_offset], 1.0e-6f);\
						delta_o[(l_o+8+nb_class+nb_param+0)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+nb_param+0)*f_offset] - TC_scale_factor*y_param.angle_sm[0]*obb_loss_scale*0.5f*obb_grad_theta*(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]/obb_theta_norm2);\
						delta_o[(l_o+8+nb_class+nb_param+1)*f_offset] = (type)((float)delta_o[(l_o+8+nb_class+nb_param+1)*f_offset] + TC_scale_factor*y_param.angle_sm[0]*obb_loss_scale*0.5f*obb_grad_theta*(float)output[(l_o+8+nb_class+nb_param+0)*f_offset]/obb_theta_norm2);\
					}																																	\
					if(multi_pos_topk > 1 && nb_im_iter > rand_startup && class_only_IoU < -1.9f)														\
					{																																	\
						multi_pos_iou_thr = fmaxf(multi_pos_min_iou, max_IoU*multi_pos_iou_ratio);														\
						for(secondary_iter = 1; secondary_iter < multi_pos_topk; secondary_iter++)														\
						{																																\
							secondary_box = -1;																											\
							best_secondary_iou = multi_pos_iou_thr;																						\
							for(k = 0; k < nb_box; k++)																									\
							{																															\
								if(box_locked[k] == 2 || dist_prior[resp_targ_offset + k] >= -1.0f)														\
									continue;																											\
								secondary_iou = IoU_table[resp_targ_offset + k];																		\
								if(secondary_iou > best_secondary_iou)																					\
								{																														\
									best_secondary_iou = secondary_iou;																					\
									secondary_box = k;																									\
								}																														\
							}																															\
							if(secondary_box < 0)																										\
								break;																													\
							secondary_iou = best_secondary_iou;																						\
							if(secondary_iou > 0.98f)																									\
								secondary_iou = 0.98f;																									\
							l_o = secondary_box*output_offset;																							\
							for(k = 0; k < 6; k++)																										\
								delta_o[(l_o+k)*f_offset] = (type)0.0f;																				\
							for(k = 0; k < nb_class; k++)																								\
								delta_o[(l_o+8+k)*f_offset] = (type)0.0f;																				\
							for(k = 0; k < nb_param; k++)																								\
								delta_o[(l_o+8+nb_class+k)*f_offset] = (type)0.0f;																		\
							for(k = 0; k < nb_angle; k++)																								\
								delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] = (type)0.0f;															\
							switch(fit_prob)																											\
							{																															\
								case 1:																													\
									if(secondary_iou > min_prob_IoU_lim)																				\
									{																													\
										prob_target = gpu_yolo_probability_quality_target(output, target, l_o, l_t, f_offset, nb_class, nb_param, nb_angle, y_param, secondary_iou, obj_in_offset);	\
										delta_o[(l_o+6)*f_offset] = (type)(TC_scale_factor*multi_pos_obj_weight*sm_tab[2][0]							\
											*prob_scale*(float)output[(l_o+6)*f_offset]																	\
											*(1.0f-(float)output[(l_o+6)*f_offset])																		\
											*((float)output[(l_o+6)*f_offset]-prob_target));															\
									}																													\
									break;																												\
								case 0:																													\
									delta_o[(l_o+6)*f_offset] = (type)(TC_scale_factor*multi_pos_obj_weight*sm_tab[2][0]								\
										*prob_scale*(float)output[(l_o+6)*f_offset]																		\
										*(1.0f-(float)output[(l_o+6)*f_offset])																			\
										*((float)output[(l_o+6)*f_offset]-0.5f));																		\
									break;																												\
							}																															\
							switch(fit_obj)																												\
							{																															\
								case 1:																													\
									if(secondary_iou > min_obj_IoU_lim)																				\
									{																													\
										obj_target = gpu_yolo_objectness_quality_target(output, target, l_o, l_t, f_offset, nb_class, nb_param, y_param, secondary_iou, obj_in_offset);	\
										delta_o[(l_o+7)*f_offset] = (type)(TC_scale_factor*multi_pos_obj_weight*sm_tab[3][0]							\
											*obj_scale*(float)output[(l_o+7)*f_offset]																	\
											*(1.0f-(float)output[(l_o+7)*f_offset])																		\
											*((float)output[(l_o+7)*f_offset]-obj_target));															\
									}																													\
									break;																												\
								case 0:																													\
									delta_o[(l_o+7)*f_offset] = (type)(TC_scale_factor*multi_pos_obj_weight*sm_tab[3][0]								\
										*obj_scale*(float)output[(l_o+7)*f_offset]																		\
										*(1.0f-(float)output[(l_o+7)*f_offset])																			\
										*((float)output[(l_o+7)*f_offset]-0.5f));																		\
									break;																												\
							}																															\
							box_locked[secondary_box] = 2;																								\
							for(j = 0; j < nb_in_cell; j++)																							\
								IoU_table[j*nb_box + secondary_box] = -2.0f;																			\
						}																																\
					}																																	\
					for(k = 0; k < nb_box; k++)																										\
						IoU_table[resp_targ_offset + k] = -2.0f;																						\
				}																																			\
	for(j = 0; j < nb_box; j++)																													\
	{																																			\
		/* If no match only update Objectness toward 0 */																						\
		/* (here it means error compute)! (no coordinate nor class update) */																	\
		l_o = j*output_offset;																													\
		if(box_locked[j] != 2)																													\
		{																																		\
			for(k = 0; k < 6; k++)																												\
				delta_o[(l_o+k)*f_offset] = (type) 0.0f;																						\
																																				\
			if(box_locked[j] == 1)																												\
			{																																	\
				delta_o[(l_o+6)*f_offset] = (type) 0.0f;																						\
				delta_o[(l_o+7)*f_offset] = (type) 0.0f;																						\
			}																																	\
			else																																\
			{																																	\
				switch(fit_prob)																												\
				{																																\
					case 1:																														\
						delta_o[(l_o+6)*f_offset] = (type)(																						\
							TC_scale_factor*sm_tab[2][0]*(lambda_noobj_prior[j])																\
							*prob_scale*(float)output[(l_o+6)*f_offset]																			\
							*(1.0f-(float)output[(l_o+6)*f_offset])																				\
							*((float)output[(l_o+6)*f_offset]-y_param.prob_quality_floor));													\
						break;																													\
					case 0:																														\
						delta_o[(l_o+6)*f_offset] = (type)(																						\
							TC_scale_factor*sm_tab[2][0]*(lambda_noobj_prior[j])																\
							*prob_scale*(float)output[(l_o+6)*f_offset]																			\
							*(1.0f-(float)output[(l_o+6)*f_offset])																				\
							*((float)output[(l_o+6)*f_offset]-0.5f));																			\
						break;																													\
					case -1:																													\
						delta_o[(l_o+6)*f_offset] = (type)(0.0f);																				\
						break;																													\
				}																																\
				switch(fit_obj)																													\
				{																																\
					case 1:																														\
						delta_o[(l_o+7)*f_offset] = (type)(																						\
							TC_scale_factor*sm_tab[3][0]*(lambda_noobj_prior[j])																\
							*obj_scale*(float)output[(l_o+7)*f_offset]																			\
							*(1.0f-(float)output[(l_o+7)*f_offset])																				\
							*((float)output[(l_o+7)*f_offset]-0.02f));																			\
						break;																													\
					case 0:																														\
						delta_o[(l_o+7)*f_offset] = (type)(																						\
							TC_scale_factor*sm_tab[3][0]*(lambda_noobj_prior[j])																\
							*obj_scale*(float)output[(l_o+7)*f_offset]																			\
							*(1.0f-(float)output[(l_o+7)*f_offset])																				\
							*((float)output[(l_o+7)*f_offset]-0.5f));																			\
						break;																													\
					case -1:																													\
						delta_o[(l_o+7)*f_offset] = (type)(0.0f);																				\
						break;																													\
				}																																\
			}																																	\
																																				\
			for(k = 0; k < nb_class; k++)																										\
				delta_o[(l_o+8+k)*f_offset] = (type) (0.0f);																					\
																																				\
				for(k = 0; k < nb_param; k++)																										\
					delta_o[(l_o+8+nb_class+k)*f_offset] = (type) (0.0f);																			\
																																					\
				for(k = 0; k < nb_angle; k++)																										\
					delta_o[(l_o+8+nb_class+nb_param+k)*f_offset] = (type) (0.0f);																	\
			}																																		\
		}																																			\
	}


#define YOLO_error_kernel(name, type)																											\
__global__ void YOLO_error_kernel_##name																										\
	(float *output_error, void *i_output, void *i_target, int flat_target_size, int flat_output_size, 											\
	int nb_area_w, int nb_area_h, int nb_area_d, yolo_param y_param, size_t size)																\
{																																				\
	int i = blockIdx.x*blockDim.x + threadIdx.x;																								\
	if(i >= size)																																\
		return;																																	\
																																				\
	type *output = (type*) i_output;																											\
	type *target = (type*) i_target;																											\
																																				\
	/* Define many "shorts" for y_param content to enhance code redeability*/																	\
	int nb_box                      = y_param.nb_box;																							\
	int nb_class                    = y_param.nb_class;																							\
	int nb_param                    = y_param.nb_param; 																						\
	int nb_angle                    = y_param.nb_angle; 																						\
	int strict_box_size_association = y_param.strict_box_size_association;																		\
	int fit_dim                     = y_param.fit_dim;																							\
	float min_prior_forced_scaling  = y_param.min_prior_forced_scaling;																			\
	int class_softmax               = y_param.class_softmax;																					\
		int diff_flag                   = y_param.diff_flag;																						\
		int prior_dist_type             = y_param.prior_dist_type;																					\
		int error_type                  = y_param.error_type;																						\
		int target_box_mode             = y_param.target_box_mode;																					\
		int obb_loss_mode               = y_param.obb_loss_mode;																					\
																																				\
	float coord_scale = y_param.scale_tab[0], size_scale  = y_param.scale_tab[1];																\
	float prob_scale  = y_param.scale_tab[2], obj_scale   = y_param.scale_tab[3];																\
	float class_scale = y_param.scale_tab[4], param_scale = y_param.scale_tab[5];																\
		float angle_scale = y_param.angle_scale, angle_unit_norm_scale = y_param.angle_unit_norm_scale;												\
		float obb_loss_scale = y_param.obb_loss_scale;																								\
																																				\
	float *prior_size         = y_param.prior_size;																								\
	int   *cell_size          = y_param.cell_size;																								\
	float *lambda_noobj_prior = y_param.noobj_prob_prior;																						\
	float **sm_tab            = y_param.slopes_and_maxes_tab;																					\
	float *param_ind_scale    = y_param.param_ind_scale;																						\
	float *IoU_monitor        = y_param.IoU_monitor;																							\
	int   *target_cell_mask   = y_param.target_cell_mask;																						\
	float *IoU_table          = y_param.IoU_table;																								\
	float *dist_prior         = y_param.dist_prior;																								\
	int   *box_locked         = y_param.box_locked;																								\
	float *box_in_pix         = y_param.box_in_pix;																								\
																																				\
	float size_max_sat = expf(sm_tab[1][1]), size_min_sat = expf(sm_tab[1][2]);																	\
	float good_IoU_lim      = y_param.IoU_limits[0], low_IoU_best_box_assoc = y_param.IoU_limits[1];											\
	float min_prob_IoU_lim  = y_param.IoU_limits[2], min_obj_IoU_lim        = y_param.IoU_limits[3];											\
	float min_class_IoU_lim = y_param.IoU_limits[4], min_param_IoU_lim      = y_param.IoU_limits[5];											\
	float min_angle_IoU_lim = y_param.min_angle_IoU_lim;																						\
	float diff_IoU_lim      = y_param.IoU_limits[6], diff_obj_lim           = y_param.IoU_limits[7];											\
	int fit_pos = y_param.fit_parts[0], fit_size  = y_param.fit_parts[1], fit_prob  = y_param.fit_parts[2]; 									\
	int fit_obj = y_param.fit_parts[3], fit_class = y_param.fit_parts[4], fit_param = y_param.fit_parts[5];										\
		int fit_angle = y_param.fit_angle;																											\
		int angle_loss_mode = y_param.angle_loss_mode;																								\
																																				\
	int j, k, l, l_o, l_t;																														\
	size_t f_offset, c_total_nb_area, c_total_nb_area_batch, total_cell_pos_nb_area, total_area_and_cell_offset;								\
	int c_batch, output_offset, target_offset, nb_obj_target, s_p_i = 0;																		\
	int nb_in_cell, id_in_cell, id_in_cell_offset, resp_box = -1, resp_targ = -1, resp_targ_offset, targ_diff_flag = 0;							\
	float best_dist, c_dist, max_IoU, current_IoU, prob_target, obj_target;																	\
	int cell_pos[3], c_nb_area[3], obj_c[3];																									\
	float *c_box_in_pix, *c_prior_size;																											\
		float obj_in_offset[6], out_int[6], targ_int[6], targ_size[3];																				\
		float obb_grad_logw, obb_grad_logh, obb_grad_theta;																							\
		float obb_pred_theta, obb_targ_theta, obb_targ_cx, obb_targ_cy, obb_loss_val;																\
		float angle_weight, angle_norm, angle_norm_diff, angle_unit_err_share;																		\
		float angle_out0, angle_out1, angle_targ0, angle_targ1, angle_dot, angle_pair_loss;															\
	float class_only_IoU = -2.0f;																												\
																																				\
	c_nb_area[0] = nb_area_w; c_nb_area[1] = nb_area_h; c_nb_area[2] = nb_area_d;																\
	c_total_nb_area = c_nb_area[0]*c_nb_area[1]*c_nb_area[2];																					\
	c_batch = i / flat_output_size;																												\
	target += flat_target_size * c_batch;																										\
	f_offset = size;																															\
	output_offset = 8+nb_class+nb_param+nb_angle;																								\
	target_offset = 7+nb_param+((nb_angle > 0) ? nb_angle + 1 : 0)+diff_flag;																	\
																																				\
	i = i % flat_output_size;																													\
	cell_pos[2] = i / (c_nb_area[0]*c_nb_area[1]);																								\
	cell_pos[1] = (int)(i % (c_nb_area[0]*c_nb_area[1])) / c_nb_area[0];																		\
	cell_pos[0] = (int)(i % (c_nb_area[0]*c_nb_area[1])) % c_nb_area[0];																		\
																																				\
	c_total_nb_area_batch = c_total_nb_area * c_batch;																							\
	total_cell_pos_nb_area = cell_pos[2]*c_nb_area[0]*c_nb_area[1] + cell_pos[1]*c_nb_area[0] + cell_pos[0];									\
	total_area_and_cell_offset = c_total_nb_area_batch + total_cell_pos_nb_area;																\
																																				\
	output_error += total_area_and_cell_offset;																									\
	output += total_area_and_cell_offset;																										\
																																				\
	IoU_monitor += 2 * nb_box * total_area_and_cell_offset;																						\
	target_cell_mask +=	total_area_and_cell_offset * y_param.max_nb_obj_per_image;																\
	/*Could redume memory footprint with a max_nb_obj_per_cell parameter*/																		\
	IoU_table  += total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;															\
	dist_prior += total_area_and_cell_offset * y_param.max_nb_obj_per_image * nb_box;															\
	box_locked += total_area_and_cell_offset * nb_box;																							\
	box_in_pix += total_area_and_cell_offset * 6 * nb_box;																						\
																																				\
	nb_obj_target = target[0];																													\
	target++;																																	\
																																				\
	if(nb_obj_target == -1)																														\
	{																																			\
		nb_obj_target = 1;																														\
		class_only_IoU = good_IoU_lim; 																											\
	}																																			\
																																				\
	best_dist = 1000000000;																														\
	for(k = 0; k < nb_box; k++)																													\
	{																																			\
		box_locked[k] = 0;																														\
		c_box_in_pix = box_in_pix + k*6;																										\
		c_prior_size = prior_size + k*3;																										\
		l_o = k*output_offset;																													\
		for(l = 0; l < 3; l++)																													\
			c_box_in_pix[l] = ((float)output[(l_o+l)*f_offset] + cell_pos[l]) * cell_size[l];													\
		for(l = 0; l < 3; l++)																													\
			c_box_in_pix[l+3] = c_prior_size[l]*expf((float)output[(l_o+l+3)*f_offset]);														\
																																				\
		c_dist = sqrt(c_prior_size[0]*c_prior_size[0] 																							\
					+ c_prior_size[1]*c_prior_size[1]																							\
					+ c_prior_size[2]*c_prior_size[2]);																							\
		if(c_dist < best_dist)																													\
		{																																		\
			best_dist = c_dist;																													\
			s_p_i = k;																															\
		}																																		\
																																				\
		IoU_monitor[k*2] = -1.0f;																												\
		IoU_monitor[k*2+1] = -1.0f;																												\
	}																																			\
																																				\
	nb_in_cell = 0;																																\
	for(j = 0; j < nb_obj_target; j++)																											\
	{																																			\
		l_t = j*target_offset;																													\
			GPU_YOLO_LOAD_TARGET_BOX(target, l_t, targ_int, target_box_mode);														\
																																				\
		/* Search for targets that should be predicted by the current cell element */															\
		target_cell_mask[j] = 1;																												\
		for(l = 0; l < 3; l++)																													\
		{																																		\
			obj_c[l] = (int)( (targ_int[l+3] + targ_int[l])*0.5f / cell_size[l]);											\
			/* If target outside the current cell element, set target flag to 0*/																\
			if(obj_c[l] != cell_pos[l])																											\
				target_cell_mask[j] = 0;																										\
		}																																		\
																																				\
		if(target_cell_mask[j] == 1)																											\
			nb_in_cell++;																														\
																																				\
		/* Flag all the "Good but not best boxes" for all targets regardless of the grid element */												\
		for(k = 0; k < nb_box; k++)																												\
		{																																		\
			l_o = k*output_offset;																												\
			if(box_locked[k] != 0)																												\
				continue;																														\
			c_box_in_pix = box_in_pix + k*6;																									\
			for(l = 0; l < 6; l++)																												\
				out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];													\
																																				\
			current_IoU = GPU_YOLO_BOX_QUALITY(out_int, targ_int, output, target, l_o, l_t, f_offset, nb_class, nb_param, nb_angle, y_param);	\
			if(current_IoU > good_IoU_lim)																										\
				box_locked[k] = 1;																												\
		}																																		\
	}																																			\
																																				\
	/* For all targets in cell compute the IoU with the predictions and distances to the priors */												\
	id_in_cell = 0;																																\
	for(j = 0; j < nb_obj_target; j++)																											\
	{																																			\
		id_in_cell_offset = id_in_cell*nb_box;																									\
		if(target_cell_mask[j] == 0)																											\
			continue;																															\
																																				\
		l_t = j*target_offset;																													\
			GPU_YOLO_LOAD_TARGET_BOX(target, l_t, targ_int, target_box_mode);														\
			for(l = 0; l < 3; l++)																													\
				targ_size[l] = targ_int[l+3] - targ_int[l];																							\
																																					\
			for(k = 0; k < nb_box; k++)																												\
			{																																		\
				l_o = k*output_offset;																												\
				c_box_in_pix = box_in_pix + k*6;																									\
				for(l = 0; l < 6; l++)																												\
					out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];													\
																																					\
				current_IoU = GPU_YOLO_BOX_QUALITY(out_int, targ_int, output, target, l_o, l_t, f_offset, nb_class, nb_param, nb_angle, y_param);	\
				IoU_table[id_in_cell_offset + k] = current_IoU;																						\
				dist_prior[id_in_cell_offset + k] = -2.0f;																							\
		}																																		\
																																				\
		/* Restrict the association to the l best theoritical prior (times repetition of identical priors) */									\
		if(error_type == ERR_COMPLETE && strict_box_size_association > 0)																		\
		{																																		\
			if(prior_dist_type == DIST_IOU)																										\
				for(l = 0; l < 6; l++)																											\
					targ_int[l] = copysignf(0.5f,l-2.5f)*targ_size[l%3];																		\
																																				\
			for(k = 0; k < nb_box; k++)																											\
			{																																	\
				c_prior_size = prior_size + k*3;																								\
				switch(prior_dist_type)																											\
				{																																\
					case DIST_IOU:																												\
						for(l = 0; l < 6; l++)																									\
							out_int[l] = copysignf(0.5f,l-2.5f)*c_prior_size[l%3];																\
						dist_prior[id_in_cell_offset + k] = 1.0f - y_param.c_IoU_fct(out_int, targ_int);										\
						break;																													\
																																				\
					default:																													\
					case DIST_SIZE:																												\
						dist_prior[id_in_cell_offset + k] = sqrt(																				\
							 (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])														\
							+(targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])														\
							+(targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]));													\
						break;																													\
																																				\
					case DIST_OFFSET:																											\
						for(l = 0; l < 3; l++)																									\
						{																														\
							obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];																	\
							if(obj_in_offset[l+3] < size_min_sat)																				\
								obj_in_offset[l+3] = logf(size_min_sat);																		\
							else if(obj_in_offset[l+3] > size_max_sat)																			\
								obj_in_offset[l+3] = logf(size_max_sat);																		\
							else																												\
								obj_in_offset[l+3] = logf(obj_in_offset[l+3]);																	\
						}																														\
																																				\
						dist_prior[id_in_cell_offset + k] = 																					\
							 fabsf(obj_in_offset[3])																							\
							+fabsf(obj_in_offset[4])																							\
							+fabsf(obj_in_offset[5]);																							\
						break;																													\
				}																																\
			}																																	\
																																				\
			for(l = 0; l < strict_box_size_association; l++)																					\
			{																																	\
				best_dist = 1000000.0f;																											\
				for(k = 0; k < nb_box; k++)																										\
					if(dist_prior[id_in_cell_offset+k] > 0.0 && dist_prior[id_in_cell_offset+k] < best_dist)									\
						best_dist = dist_prior[id_in_cell_offset+k];																			\
				for(k = 0; k < nb_box; k++) /* Flag the closest theoritical prior (and identical ones if any) */								\
					if(fabsf(dist_prior[id_in_cell_offset+k] - best_dist) < 0.001f)																\
						dist_prior[id_in_cell_offset+k] = -2.0f;																				\
			}																																	\
		}																																		\
																																				\
		id_in_cell++;																															\
	}																																			\
																																				\
	for(id_in_cell = 0; id_in_cell < nb_in_cell; id_in_cell++)																					\
	{																																			\
		/* No random association in error display*/																								\
		max_IoU = -2.0f; resp_box = -1;	resp_targ = -1;																							\
		for(j = 0; j < nb_in_cell; j++)																											\
			for(k = 0; k < nb_box; k++)																											\
				if(IoU_table[j*nb_box+k] > max_IoU && dist_prior[j*nb_box+k] < -1.0f)															\
				{																																\
					max_IoU = IoU_table[j*nb_box+k];																							\
					resp_targ = j;																												\
					resp_box = k;																												\
				}																																\
																																				\
		/* If strict_box_size > 0 and no more good prior is available, or if there is more targets than boxes */								\
		/* In that case all the remaining target are unable to be associated to */ 																\
		/* any other box and the id_in_cell loop must be stoped */																				\
		if(resp_box == -1)																														\
			continue;																															\
																																				\
		/* l is the "best" index in the "in cell" list */																						\
		/*Need to get back the original target index from the "in cell" index*/																	\
		l = 0;																																	\
		for(j = 0; j < nb_obj_target; j++)																										\
		{																																		\
			l += target_cell_mask[j];																											\
			if(l == resp_targ + 1)																												\
				break;																															\
		}																																		\
		/* The appropriate j is defined after this early stop loop*/																			\
		l_t = j*target_offset;																													\
		resp_targ_offset = resp_targ*nb_box;																									\
																																				\
		if(error_type == ERR_COMPLETE)																											\
		{																																		\
			GPU_YOLO_LOAD_TARGET_BOX(target, l_t, targ_int, target_box_mode);														\
			for(l = 0; l < 3; l++)																												\
				targ_size[l] = targ_int[l+3] - targ_int[l];																						\
																																				\
			/* Force the association to the smallest prior (or identical) if the target is too small */											\
			if(    targ_size[0] < min_prior_forced_scaling*prior_size[s_p_i*3+0]																\
				&& targ_size[1] < min_prior_forced_scaling*prior_size[s_p_i*3+1]																\
				&& targ_size[2] < min_prior_forced_scaling*prior_size[s_p_i*3+2])																\
			{																																	\
				max_IoU = -2.0f; best_dist = prior_size[s_p_i*3+0]*prior_size[s_p_i*3+1]*prior_size[s_p_i*3+2];									\
				for(k = 0; k < nb_box; k++)																										\
				{																																\
					c_prior_size = prior_size + k*3;																							\
					if((   prior_size[s_p_i*3+0] == c_prior_size[0] 																			\
						&& prior_size[s_p_i*3+1] == c_prior_size[1] 																			\
						&& prior_size[s_p_i*3+2] == c_prior_size[2]) 																			\
						&& IoU_table[resp_targ_offset + k] > max_IoU)																			\
					{																															\
						max_IoU = IoU_table[resp_targ_offset + k];																				\
						resp_box = k;																											\
					}																															\
				}																																\
			}																																	\
			/* If prediction is too bad, associate it with the best theoritical prior instead (might found the same box again) */				\
			/* Also force the best theoritical prior association at a small rate */																\
			else if(max_IoU < low_IoU_best_box_assoc)																							\
			{																																	\
				if(prior_dist_type == DIST_IOU)																									\
					for(l = 0; l < 6; l++)																										\
						targ_int[l] = copysignf(0.5f,l-2.5f)*targ_size[l%3];																	\
																																				\
				best_dist = 100000.0f;																											\
				for(k = 0; k < nb_box; k++)																										\
				{																																\
					c_prior_size = prior_size + k*3;																							\
					switch(prior_dist_type)																										\
					{																															\
						case DIST_IOU:																											\
							for(l = 0; l < 6; l++)																								\
								out_int[l] = copysignf(0.5f,l-2.5f)*c_prior_size[l%3];															\
							dist_prior[resp_targ_offset + k] = 1.0f - y_param.c_IoU_fct(out_int, targ_int);										\
							break;																												\
																																				\
						default:																												\
						case DIST_SIZE:																											\
							dist_prior[resp_targ_offset + k] = sqrt(																			\
								 (targ_size[0]-c_prior_size[0])*(targ_size[0]-c_prior_size[0])													\
								+(targ_size[1]-c_prior_size[1])*(targ_size[1]-c_prior_size[1])													\
								+(targ_size[2]-c_prior_size[2])*(targ_size[2]-c_prior_size[2]));												\
							break;																												\
																																				\
						case DIST_OFFSET:																										\
							for(l = 0; l < 3; l++)																								\
							{																													\
								obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];																\
								if(obj_in_offset[l+3] < size_min_sat)																			\
									obj_in_offset[l+3] = logf(size_min_sat);																	\
								else if(obj_in_offset[l+3] > size_max_sat)																		\
									obj_in_offset[l+3] = logf(size_max_sat);																	\
								else																											\
									obj_in_offset[l+3] = logf(obj_in_offset[l+3]);																\
							}																													\
																																				\
							dist_prior[resp_targ_offset + k] =																					\
								 fabsf(obj_in_offset[3])																						\
								+fabsf(obj_in_offset[4])																						\
								+fabsf(obj_in_offset[5]);																						\
							break;																												\
					}																															\
					if(dist_prior[resp_targ_offset + k] < best_dist)																			\
						best_dist = dist_prior[resp_targ_offset + k];																			\
				}																																\
				max_IoU = -2.0f;																												\
				for(k = 0; k < nb_box; k++)																										\
				{																																\
					if(fabsf(dist_prior[resp_targ_offset + k] - best_dist) < 0.001f && IoU_table[resp_targ_offset + k] > max_IoU)				\
					{																															\
						max_IoU = IoU_table[resp_targ_offset + k];																				\
						resp_box = k;																											\
					}																															\
				}																																\
				/* If the best prior (or identical) is not available, the resp_box is unchanged */												\
				/* Should always get a resp_box != -1, regarding all previous conditions */														\
			}																																	\
		}																																		\
																																				\
		/* Mark the target as already associated by removing its contributions to the IoU table */												\
		for(k = 0; k < nb_box; k++)																												\
			IoU_table[resp_targ_offset + k] = -2.0f;																							\
																																				\
			c_box_in_pix = box_in_pix + resp_box*6;																									\
			for(l = 0; l < 6; l++)																													\
				out_int[l] = c_box_in_pix[l%3] + copysignf(0.5f,l-2.5f)*c_box_in_pix[3+l%3];														\
																																					\
			GPU_YOLO_LOAD_TARGET_BOX(target, l_t, targ_int, target_box_mode);														\
			for(l = 0; l < 3; l++)																													\
				targ_size[l] = targ_int[l+3] - targ_int[l];																							\
																																					\
			l_o = resp_box*output_offset;																											\
			max_IoU = GPU_YOLO_BOX_QUALITY(out_int, targ_int, output, target, l_o, l_t, f_offset, nb_class, nb_param, nb_angle, y_param);			\
			if(max_IoU > 0.98f)																														\
				max_IoU = 0.98f;																													\
			if(class_only_IoU > -2.0f)																												\
				max_IoU = class_only_IoU; /*regardless of actual IoU because class only box is not precise*/										\
																																					\
			c_prior_size = prior_size + 3*resp_box;																									\
																																				\
		/* Positive reinforcement */ 																											\
		targ_diff_flag = 0;																														\
		if(diff_flag)	/* Cast from mixed precision type to float is always possible, but not necessary to int directly */						\
			targ_diff_flag = (int)((float)target[l_t+7+nb_param+((nb_angle > 0) ? nb_angle + 1 : 0)]);											\
																																				\
		/* If the target is flagged as "difficult", only update the matching box if the prediction is already confident enough */				\
		/* The target is removed from the list anyway, and the corresponding box fall to "background" or "Good_but_not_best" case*/				\
		if(diff_flag && targ_diff_flag > 0																										\
			&& (error_type == ERR_NATURAL || max_IoU < diff_IoU_lim || (float)output[(l_o+7)*f_offset] < diff_obj_lim))							\
			continue;																															\
																																				\
		/* Mark the box as already associated by removing its contributions to the IoU table */													\
		for(j = 0; j < nb_in_cell; j++)																											\
			IoU_table[j*nb_box + resp_box] = -2.0f;																								\
																																				\
		box_locked[resp_box] = 2;																												\
																																				\
		IoU_monitor[resp_box*2] = (float)output[(l_o+7)*f_offset];																				\
		IoU_monitor[resp_box*2+1] = max_IoU;																									\
																																				\
		for(l = 0; l < 3; l++)																													\
			obj_in_offset[l] = ((targ_int[l+3] + targ_int[l])*0.5f - cell_pos[l]*cell_size[l])/(float)cell_size[l];								\
		for(l = 0; l < 3; l++)																													\
		{																																		\
			obj_in_offset[l+3] = targ_size[l]/c_prior_size[l];																					\
			if(obj_in_offset[l+3] < size_min_sat)																								\
				obj_in_offset[l+3] = logf(size_min_sat);																						\
			else if(obj_in_offset[l+3] > size_max_sat)																							\
				obj_in_offset[l+3] = logf(size_max_sat);																						\
			else																																\
				obj_in_offset[l+3] = logf(obj_in_offset[l+3]);																					\
		}																																		\
																																				\
		switch(fit_pos)																															\
		{																																		\
			case 1:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k && class_only_IoU < -1.9f && (diff_flag == 0 || targ_diff_flag < 3))											\
						output_error[(l_o+k)*f_offset] = 0.5f*coord_scale																		\
							*((float)output[(l_o+k)*f_offset]-obj_in_offset[k])																	\
							*((float)output[(l_o+k)*f_offset]-obj_in_offset[k]);																\
					else																														\
						output_error[(l_o+k)*f_offset] = 0.0f;																					\
				}																																\
				break;																															\
			case 0:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k)																												\
						output_error[(l_o+k)*f_offset] = 0.5f*coord_scale																		\
							*((float)output[(l_o+k)*f_offset]-0.0f)																				\
							*((float)output[(l_o+k)*f_offset]-0.0f);																			\
					else																														\
						output_error[(l_o+k)*f_offset] = 0.0f;																					\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < 3; k++)																											\
					output_error[(l_o+k)*f_offset] = 0.0f;																						\
				break;																															\
		}																																		\
																																				\
		switch(fit_size)																														\
		{																																		\
			case 1:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k && class_only_IoU < -1.9f && (diff_flag == 0 || targ_diff_flag < 3))											\
						output_error[(l_o+k+3)*f_offset] = 0.5f*size_scale																		\
						*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3])																	\
						*((float)output[(l_o+k+3)*f_offset]-obj_in_offset[k+3]);																\
					else																														\
						output_error[(l_o+k+3)*f_offset] = 0.0f;																				\
				}																																\
				break;																															\
			case 0:																																\
				for(k = 0; k < 3; k++)																											\
				{																																\
					if(fit_dim > k)																												\
						output_error[(l_o+k+3)*f_offset] = 0.5f*size_scale																		\
						*((float)output[(l_o+k+3)*f_offset]-0.0f)																				\
						*((float)output[(l_o+k+3)*f_offset]-0.0f);																				\
					else																														\
						output_error[(l_o+k+3)*f_offset] = 0.0f;																				\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < 3; k++)																											\
					output_error[(l_o+k+3)*f_offset] = 0.0f;																					\
				break;																															\
		}																																		\
																																				\
		switch(fit_prob)																														\
		{																																		\
			case 1:																																\
				if(max_IoU > min_prob_IoU_lim || error_type == ERR_NATURAL)																		\
				{																																\
					prob_target = gpu_yolo_probability_quality_target(output, target, l_o, l_t, f_offset, nb_class, nb_param, nb_angle, y_param, max_IoU, obj_in_offset);	\
					output_error[(l_o+6)*f_offset] = 0.5f*prob_scale																			\
						*((float)output[(l_o+6)*f_offset]-prob_target)																			\
						*((float)output[(l_o+6)*f_offset]-prob_target);																		\
				}																																\
				else																															\
					output_error[(l_o+6)*f_offset] = 0.0f;																						\
				break;																															\
			case 0:																																\
				output_error[(l_o+6)*f_offset] = 0.5f*prob_scale																				\
					*((float)output[(l_o+6)*f_offset]-0.5f)																						\
					*((float)output[(l_o+6)*f_offset]-0.5f);																					\
				break;																															\
			case -1:																															\
				output_error[(l_o+6)*f_offset] = 0.0f;																							\
				break;																															\
		}																																		\
																																				\
		switch(fit_obj)																															\
		{																																		\
			case 1:																																\
				if(max_IoU > min_obj_IoU_lim || error_type == ERR_NATURAL)																		\
				{																																\
					obj_target = gpu_yolo_objectness_quality_target(output, target, l_o, l_t, f_offset, nb_class, nb_param, y_param, max_IoU, obj_in_offset);	\
					output_error[(l_o+7)*f_offset] = 0.5f*obj_scale																				\
						*((float)output[(l_o+7)*f_offset]-obj_target)																			\
						*((float)output[(l_o+7)*f_offset]-obj_target);																			\
				}																																\
				else																															\
					output_error[(l_o+7)*f_offset] = 0.0f;																						\
				break;																															\
			case 0:																																\
				output_error[(l_o+7)*f_offset] = 0.5f*obj_scale																					\
					*((float)output[(l_o+7)*f_offset]-0.5)																						\
					*((float)output[(l_o+7)*f_offset]-0.5);																						\
				break;																															\
			case -1:																															\
				output_error[(l_o+7)*f_offset] = 0.0f;																							\
				break;																															\
		}																																		\
																																				\
		/*Note : mean square error on classes => could be changed to soft max but difficult to balance*/										\
		switch(fit_class)																														\
		{																																		\
			case 1:																																\
				if((max_IoU > min_class_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2)) || error_type == ERR_NATURAL)						\
				{																																\
					if(class_softmax)																											\
					{																															\
						for(k = 0; k < nb_class; k++)																							\
						{																														\
							if(k == (int)target[l_t]-1)																							\
							{																													\
								if((float)output[(l_o+8+k)*f_offset] > 0.0000001f)																\
									output_error[(l_o+8+k)*f_offset] = class_scale																\
										*(-logf((float)output[(l_o+8+k)*f_offset]));															\
								else																											\
									output_error[(l_o+8+k)*f_offset] = class_scale*(-logf(0.0000001f));											\
							}																													\
							else																												\
								output_error[(l_o+8+k)*f_offset] = 0.0f;																		\
						}																														\
					}																															\
					else																														\
					{																															\
						for(k = 0; k < nb_class; k++)																							\
						{																														\
							if(k == (int)target[l_t]-1)																							\
								output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale																\
									*((float)output[(l_o+8+k)*f_offset]-0.98f)																	\
									*((float)output[(l_o+8+k)*f_offset]-0.98f);																	\
							else																												\
								output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale																\
									*((float)output[(l_o+8+k)*f_offset]-0.02f)																	\
									*((float)output[(l_o+8+k)*f_offset]-0.02f);																	\
						}																														\
					}																															\
				}																																\
				else																															\
					for(k = 0; k < nb_class; k++)																								\
						output_error[(l_o+8+k)*f_offset] = 0.0f;																				\
				break;																															\
			case 0:																																\
				if(class_softmax)																												\
				{																																\
					/* Could compute CE with target = 1/nb_class, but in this case perfect classification error > 0 (still minimum) */			\
					for(k = 0; k < nb_class; k++)																								\
						output_error[(l_o+8+k)*f_offset] = 0.0f;																				\
				}																																\
				else																															\
				{																																\
					for(k = 0; k < nb_class; k++)																								\
						output_error[(l_o+8+k)*f_offset] = 0.5f*class_scale																		\
							*((float)output[(l_o+8+k)*f_offset]-0.5f)																			\
							*((float)output[(l_o+8+k)*f_offset]-0.5f);																			\
				}																																\
				break;																															\
			case -1:																															\
				for(k = 0; k < nb_class; k++)																									\
					output_error[(l_o+8+k)*f_offset] = 0.0f;																					\
				break;																															\
		}																																		\
																																				\
		/*Linear error of additional parameters*/																								\
			switch(fit_param)																														\
			{																																		\
				case 1:																																\
					if((max_IoU > min_param_IoU_lim && (diff_flag == 0 || targ_diff_flag < 2)) || error_type == ERR_NATURAL)						\
					for(k = 0; k < nb_param; k++)																								\
					{																															\
						if(gpu_yolo_flux_refine_is_aux_param(y_param, k, nb_param))																\
						{																														\
							output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;																	\
							continue;																											\
						}																														\
						output_error[(l_o+8+nb_class+k)*f_offset] = (param_ind_scale[k]*0.5f*param_scale										\
							*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k])												\
							*((float)output[(l_o+8+nb_class+k)*f_offset]-(float)target[l_t+7+k]));												\
					}																															\
				else																															\
					for(k = 0; k < nb_param; k++)																								\
						output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;																		\
				break;																															\
			case 0:																																\
				for(k = 0; k < nb_param; k++)																									\
				{																																\
					if(gpu_yolo_flux_refine_is_aux_param(y_param, k, nb_param))																	\
					{																															\
						output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;																		\
						continue;																												\
					}																															\
					output_error[(l_o+8+nb_class+k)*f_offset] = (param_ind_scale[k]*0.5f*param_scale											\
						*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f)																		\
						*((float)output[(l_o+8+nb_class+k)*f_offset]-0.5f));																	\
				}																																\
				break;																															\
			default:																															\
				case -1:																															\
					for(k = 0; k < nb_param; k++)																									\
						output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;																			\
					break;																															\
			}																																		\
																																					\
			/*Encoded angle head display error*/																									\
			switch(fit_angle)																														\
			{																																		\
				case 1:																																\
					if(nb_angle > 0 && ((max_IoU > min_angle_IoU_lim && (diff_flag == 0 || targ_diff_flag < 3)) || error_type == ERR_NATURAL))		\
					{																																\
						angle_weight = (float)target[l_t+7+nb_param+nb_angle];																		\
						if(angle_loss_mode == 1)																									\
						{																															\
							for(k = 0; k < nb_angle; k += 2)																						\
							{																														\
								if(k + 1 >= nb_angle)																								\
								{																													\
									output_error[(l_o+8+nb_class+nb_param+k)*f_offset] =															\
										(angle_weight*0.5f*angle_scale																				\
										*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k])						\
										*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k]));					\
									continue;																										\
								}																													\
								angle_out0 = (float)output[(l_o+8+nb_class+nb_param+k)*f_offset];													\
								angle_out1 = (float)output[(l_o+8+nb_class+nb_param+k+1)*f_offset];												\
								angle_targ0 = (float)target[l_t+7+nb_param+k];																		\
								angle_targ1 = (float)target[l_t+7+nb_param+k+1];																	\
								angle_norm = sqrtf(angle_out0*angle_out0 + angle_out1*angle_out1);													\
								angle_norm = fmaxf(angle_norm, 1.0e-8f);																			\
								angle_dot = (angle_out0*angle_targ0 + angle_out1*angle_targ1) / angle_norm;										\
								angle_dot = fminf(1.0f, fmaxf(-1.0f, angle_dot));																	\
								angle_pair_loss = angle_weight*angle_scale*(1.0f - angle_dot);														\
								angle_unit_err_share = 0.0f;																						\
								if(angle_unit_norm_scale > 0.0f)																					\
								{																													\
									angle_norm_diff = angle_norm - 1.0f;																			\
									angle_unit_err_share = 0.25f*angle_unit_norm_scale*angle_norm_diff*angle_norm_diff;							\
								}																													\
								output_error[(l_o+8+nb_class+nb_param+k)*f_offset] = 0.5f*angle_pair_loss + angle_unit_err_share;					\
								output_error[(l_o+8+nb_class+nb_param+k+1)*f_offset] = 0.5f*angle_pair_loss + angle_unit_err_share;				\
							}																														\
						}																															\
						else																														\
						{																															\
							angle_unit_err_share = 0.0f;																							\
							if(nb_angle == 2 && angle_unit_norm_scale > 0.0f)																		\
							{																														\
								angle_norm = sqrtf(																									\
									(float)output[(l_o+8+nb_class+nb_param+0)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+0)*f_offset]			\
									+(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]*(float)output[(l_o+8+nb_class+nb_param+1)*f_offset]);		\
								angle_norm_diff = angle_norm - 1.0f;																				\
								angle_unit_err_share = 0.25f*angle_unit_norm_scale*angle_norm_diff*angle_norm_diff;								\
							}																														\
							for(k = 0; k < nb_angle; k++)																							\
								output_error[(l_o+8+nb_class+nb_param+k)*f_offset] =																\
									(angle_weight*0.5f*angle_scale																					\
									*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k])							\
									*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]-(float)target[l_t+7+nb_param+k])							\
									+angle_unit_err_share);																						\
						}																															\
					}																																\
					else																															\
						for(k = 0; k < nb_angle; k++)																								\
							output_error[(l_o+8+nb_class+nb_param+k)*f_offset] = 0.0f;																\
					break;																															\
				case 0:																																\
					for(k = 0; k < nb_angle; k++)																									\
						output_error[(l_o+8+nb_class+nb_param+k)*f_offset] = 0.5f*angle_scale														\
							*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset])																	\
							*((float)output[(l_o+8+nb_class+nb_param+k)*f_offset]);																	\
					break;																															\
				case -1:																															\
					for(k = 0; k < nb_angle; k++)																									\
						output_error[(l_o+8+nb_class+nb_param+k)*f_offset] = 0.0f;																	\
					break;																															\
				}																																		\
				if(obb_loss_mode > 0 && obb_loss_scale > 0.0f && nb_angle >= 2																		\
					&& ((max_IoU > min_angle_IoU_lim && (diff_flag == 0 || targ_diff_flag < 3)) || error_type == ERR_NATURAL))						\
				{																																	\
					obb_pred_theta = 0.5f*atan2f((float)output[(l_o+8+nb_class+nb_param+1)*f_offset], (float)output[(l_o+8+nb_class+nb_param+0)*f_offset]);\
					obb_targ_theta = 0.5f*atan2f((float)target[l_t+7+nb_param+1], (float)target[l_t+7+nb_param+0]);								\
					obb_targ_cx = 0.5f*(targ_int[0] + targ_int[3]);																				\
					obb_targ_cy = 0.5f*(targ_int[1] + targ_int[4]);																				\
					obb_loss_val = 0.5f*obb_loss_scale*gpu_yolo_obb_cov_loss_terms(c_box_in_pix[0], c_box_in_pix[1], c_box_in_pix[3], c_box_in_pix[4], obb_pred_theta, obb_targ_cx, obb_targ_cy, targ_size[0], targ_size[1], obb_targ_theta, &obb_grad_logw, &obb_grad_logh, &obb_grad_theta);\
					output_error[(l_o+0)*f_offset] += 0.25f*obb_loss_val;																			\
					output_error[(l_o+1)*f_offset] += 0.25f*obb_loss_val;																			\
					output_error[(l_o+3)*f_offset] += 0.25f*obb_loss_val;																			\
					output_error[(l_o+4)*f_offset] += 0.25f*obb_loss_val;																			\
					output_error[(l_o+8+nb_class+nb_param+0)*f_offset] += 0.25f*obb_loss_val;														\
					output_error[(l_o+8+nb_class+nb_param+1)*f_offset] += 0.25f*obb_loss_val;														\
				}																																	\
			}																																			\
	for(j = 0; j < nb_box; j++)																													\
	{																																			\
		/*If no match only update Objectness toward 0 */																						\
		/*(here it means error compute)! (no coordinate nor class update)*/																		\
		l_o = j*output_offset;																													\
		if(box_locked[j] != 2)																													\
		{																																		\
			for(k = 0; k < 6; k++)																												\
				output_error[(l_o+k)*f_offset] = 0.0f;																							\
																																				\
			if(box_locked[j] == 1)																												\
			{																																	\
				output_error[(l_o+6)*f_offset] = 0.0f;																							\
				output_error[(l_o+7)*f_offset] = 0.0f;																							\
			}																																	\
			else																																\
			{																																	\
				switch(fit_prob)																												\
				{																																\
					case 1:																														\
						output_error[(l_o+6)*f_offset] = 0.5f*(lambda_noobj_prior[j])*prob_scale												\
							*((float)output[(l_o+6)*f_offset]-y_param.prob_quality_floor)														\
							*((float)output[(l_o+6)*f_offset]-y_param.prob_quality_floor);														\
						break;																													\
					case 0:																														\
						output_error[(l_o+6)*f_offset] = 0.5f*(lambda_noobj_prior[j])*prob_scale												\
							*((float)output[(l_o+6)*f_offset]-0.5f)																				\
							*((float)output[(l_o+6)*f_offset]-0.5f);																			\
						break;																													\
					case -1:																													\
						output_error[(l_o+6)*f_offset] = 0.0f;																					\
						break;																													\
				}																																\
																																				\
				switch(fit_obj)																													\
				{																																\
					case 1:																														\
						output_error[(l_o+7)*f_offset] = 0.5f*(lambda_noobj_prior[j])*obj_scale													\
							*((float)output[(l_o+7)*f_offset]-0.02f)																			\
							*((float)output[(l_o+7)*f_offset]-0.02f);																			\
						break;																													\
					case 0:																														\
						output_error[(l_o+7)*f_offset] = 0.5f*(lambda_noobj_prior[j])*obj_scale													\
							*((float)output[(l_o+7)*f_offset]-0.5f)																				\
							*((float)output[(l_o+7)*f_offset]-0.5f);																			\
						break;																													\
					case -1:																													\
						output_error[(l_o+7)*f_offset] = 0.0f;																					\
						break;																													\
				}																																\
			}																																	\
																																				\
			for(k = 0; k < nb_class; k++)																										\
				output_error[(l_o+8+k)*f_offset] = 0.0f;																						\
																																				\
				for(k = 0; k < nb_param; k++)																										\
					output_error[(l_o+8+nb_class+k)*f_offset] = 0.0f;																				\
																																					\
				for(k = 0; k < nb_angle; k++)																										\
					output_error[(l_o+8+nb_class+nb_param+k)*f_offset] = 0.0f;																		\
																																					\
			}																																		\
		}																																			\
	}


#define typed_cuda_activ_fct_association(name)																									\
void typed_cuda_activ_fct_association_##name(network *net)																						\
{																																				\
	net->cu_inst.cu_linear_activ_fcts.activ_fct = linear_activation_kernel_##name;																\
	net->cu_inst.cu_linear_activ_fcts.deriv_fct = linear_deriv_kernel_##name;																	\
	net->cu_inst.cu_linear_activ_fcts.deriv_output_error_fct = quadratic_deriv_output_error_kernel_##name;										\
	net->cu_inst.cu_linear_activ_fcts.output_error_fct = quadratic_output_error_kernel_##name;													\
																																				\
	net->cu_inst.cu_ReLU_activ_fcts.activ_fct = ReLU_activation_kernel_##name;																	\
	net->cu_inst.cu_ReLU_activ_fcts.deriv_fct = ReLU_deriv_kernel_##name;																		\
	net->cu_inst.cu_ReLU_activ_fcts.deriv_output_error_fct = quadratic_deriv_output_error_kernel_##name;										\
	net->cu_inst.cu_ReLU_activ_fcts.output_error_fct = quadratic_output_error_kernel_##name; 													\
																																				\
	net->cu_inst.cu_logistic_activ_fcts.activ_fct = logistic_activation_kernel_##name;															\
	net->cu_inst.cu_logistic_activ_fcts.deriv_fct = logistic_deriv_kernel_##name;																\
	net->cu_inst.cu_logistic_activ_fcts.deriv_output_error_fct = quadratic_deriv_output_error_kernel_##name;									\
	net->cu_inst.cu_logistic_activ_fcts.output_error_fct = quadratic_output_error_kernel_##name;												\
																																				\
	net->cu_inst.cu_softmax_activ_fcts.activ_fct = softmax_activation_kernel_##name;															\
	net->cu_inst.cu_softmax_activ_fcts.deriv_output_error_fct = cross_entropy_deriv_output_error_kernel_##name;									\
	net->cu_inst.cu_softmax_activ_fcts.output_error_fct = cross_entropy_output_error_kernel_##name;												\
																																				\
	net->cu_inst.cu_YOLO_activ_fcts.activ_fct = YOLO_activation_kernel_##name;																	\
	net->cu_inst.cu_YOLO_activ_fcts.deriv_output_error_fct = YOLO_deriv_error_kernel_##name;													\
	net->cu_inst.cu_YOLO_activ_fcts.output_error_fct = YOLO_error_kernel_##name;																\
																																				\
}

linear_activation_kernel(FP32, float);
linear_deriv_kernel(FP32, float);
ReLU_activation_kernel(FP32, float);
ReLU_deriv_kernel(FP32, float);
quadratic_deriv_output_error_kernel(FP32, float);
quadratic_output_error_kernel(FP32, float);
logistic_activation_kernel(FP32, float, expf);
logistic_deriv_kernel(FP32, float);
softmax_activation_kernel(FP32, float, expf);
cross_entropy_deriv_output_error_kernel(FP32, float);
cross_entropy_output_error_kernel(FP32, float);
YOLO_activation_kernel(FP32, float, expf);
YOLO_deriv_error_kernel(FP32, float);
YOLO_error_kernel(FP32, float);
typed_cuda_activ_fct_association(FP32);


#if defined(GEN_VOLTA) || defined(GEN_AMPERE) 
linear_activation_kernel(FP16, half);
linear_deriv_kernel(FP16, half);
ReLU_activation_kernel(FP16, half);
ReLU_deriv_kernel(FP16, half);
quadratic_deriv_output_error_kernel(FP16, half);
quadratic_output_error_kernel(FP16, half);
logistic_activation_kernel(FP16, half, expf);
logistic_deriv_kernel(FP16, half);
softmax_activation_kernel(FP16, half, expf);
cross_entropy_deriv_output_error_kernel(FP16, half);
cross_entropy_output_error_kernel(FP16, half);
YOLO_activation_kernel(FP16, half, expf);
YOLO_deriv_error_kernel(FP16, half);
YOLO_error_kernel(FP16, half);
typed_cuda_activ_fct_association(FP16);
#endif


#if defined(GEN_AMPERE) 
linear_activation_kernel(BF16, nv_bfloat16);
linear_deriv_kernel(BF16, nv_bfloat16);
ReLU_activation_kernel(BF16, nv_bfloat16);
ReLU_deriv_kernel(BF16, nv_bfloat16);
quadratic_deriv_output_error_kernel(BF16, nv_bfloat16);
quadratic_output_error_kernel(BF16, nv_bfloat16);
logistic_activation_kernel(BF16, nv_bfloat16, expf);
logistic_deriv_kernel(BF16, nv_bfloat16);
softmax_activation_kernel(BF16, nv_bfloat16, expf);
cross_entropy_deriv_output_error_kernel(BF16, nv_bfloat16);
cross_entropy_output_error_kernel(BF16, nv_bfloat16);
YOLO_activation_kernel(BF16, nv_bfloat16, expf);
YOLO_deriv_error_kernel(BF16, nv_bfloat16);
YOLO_error_kernel(BF16, nv_bfloat16);
typed_cuda_activ_fct_association(BF16);
#endif


//#####################################################
//		 Linear activation related functions
//#####################################################

void cuda_linear_activation(layer *current)
{
	linear_param *param = (linear_param*)current->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_linear_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, param->dim, param->biased_dim, 
		param->offset, current->c_network->length, param->size);
}


void cuda_linear_deriv(layer *previous)
{
	linear_param *param = (linear_param*)previous->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	previous->c_network->cu_inst.cu_linear_activ_fcts.deriv_fct<<< cu_blocks, cu_threads >>>
		(previous->delta_o, param->dim, param->biased_dim, 
		param->offset, previous->c_network->length, param->size);
}


void cuda_linear_deriv_output_error(layer *current)
{	
	linear_param *param = (linear_param*)current->activ_param;
	
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_linear_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->target,
		param->dim, param->biased_dim, param->offset, current->c_network->length, 
		param->size, current->c_network->TC_scale_factor);
}


void cuda_linear_output_error(layer *current)
{	
	linear_param *param = (linear_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_linear_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, current->c_network->target, 
		param->dim, param->biased_dim, param->offset, current->c_network->length, param->size);
}


//#####################################################
//		 ReLU activation related functions
//#####################################################

void cuda_ReLU_activation(layer *current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, param->dim, param->biased_dim, param->offset, param->saturation, 
		param->leaking_factor, current->c_network->length, param->size);
}


void cuda_ReLU_deriv(layer *previous)
{
	ReLU_param *param = (ReLU_param*)previous->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	previous->c_network->cu_inst.cu_ReLU_activ_fcts.deriv_fct<<< cu_blocks, cu_threads >>>
		(previous->delta_o, previous->output, param->dim, param->biased_dim, param->offset, 
		param->saturation, param->leaking_factor, previous->c_network->length, param->size);
}


// Should re write an output function to take into account ReLU for Conv output format
void cuda_ReLU_deriv_output_error(layer* current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;
	cu_blocks = ( param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->target, param->dim, param->biased_dim,
		param->offset, current->c_network->length, param->size, current->c_network->TC_scale_factor);
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.deriv_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, param->dim, param->biased_dim,
		param->offset, param->saturation, param->leaking_factor, current->c_network->length, param->size);
}


void cuda_ReLU_output_error(layer* current)
{
	ReLU_param *param = (ReLU_param*)current->activ_param;	
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_ReLU_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, current->c_network->target, 
		param->dim, param->biased_dim, param->offset, current->c_network->length, param->size);
}


//#####################################################
//		 Logistic activation related functions
//#####################################################

void cuda_logistic_activation(layer *current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;

	current->c_network->cu_inst.cu_logistic_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, param->beta, param->saturation, param->dim, 
		param->biased_dim, param->offset, current->c_network->length, param->size);
}


void cuda_logistic_deriv(layer *previous)
{
	logistic_param *param = (logistic_param*)previous->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	previous->c_network->cu_inst.cu_logistic_activ_fcts.deriv_fct<<< cu_blocks, cu_threads >>>
		(previous->delta_o, previous->output, param->beta, param->dim, 
		param->biased_dim, param->offset, previous->c_network->length, param->size);
}


void cuda_logistic_deriv_output_error(layer* current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_logistic_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->target, param->dim, param->biased_dim, 
		param->offset, current->c_network->length, param->size, current->c_network->TC_scale_factor);
	
	current->c_network->cu_inst.cu_logistic_activ_fcts.deriv_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, param->beta, param->dim, 
		param->biased_dim, param->offset, current->c_network->length, param->size);
}


void cuda_logistic_output_error(layer* current)
{
	logistic_param *param = (logistic_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_logistic_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, current->c_network->target,
		param->dim, param->biased_dim, param->offset, current->c_network->length, param->size);
}

//#####################################################
//		 Softmax activation related functions
//#####################################################

void cuda_softmax_activation(layer *current)
{
	softmax_param *param = (softmax_param*)current->activ_param;
	cu_blocks = (current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_softmax_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, param->dim, param->biased_dim, param->offset, 
		current->c_network->length, current->c_network->batch_size, param->size);
}


void cuda_softmax_deriv(layer *previous)
{
	printf("ERROR: Softmax activation can not be used in the middle of the network !\n");
	exit(EXIT_FAILURE);
}


void cuda_softmax_deriv_output_error(layer *current)
{
	//use by default a cross entropy error
	softmax_param *param = (softmax_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_softmax_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->target,
		param->dim, param->biased_dim, param->offset, current->c_network->length,
		param->size, current->c_network->TC_scale_factor);
}


void cuda_softmax_output_error(layer *current)
{
	//use by default a cross entropy error
	softmax_param *param = (softmax_param*)current->activ_param;
	cu_blocks = (param->size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_softmax_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, 
		current->c_network->target, param->dim, param->biased_dim, param->offset, 
		current->c_network->length, param->size);
}

//#####################################################
//		 YOLO activation related functions
//#####################################################

void cuda_YOLO_activation(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	cu_blocks = ((size_t)current->c_network->out_size *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_YOLO_activ_fcts.activ_fct<<< cu_blocks, cu_threads >>>
		(current->output, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size,
		a_param->biased_dim*current->c_network->batch_size, *a_param, a_param->size, a_param->class_softmax);
}


void cuda_YOLO_deriv(layer *previous)
{
	printf("ERROR : YOLO activation can not be used in the middle of the network !\n");
	exit(EXIT_FAILURE);
}


void cuda_YOLO_deriv_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	cu_blocks = ((size_t)c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_YOLO_activ_fcts.deriv_output_error_fct<<< cu_blocks, cu_threads >>>
		(current->delta_o, current->output, current->c_network->target, current->c_network->output_dim, 
		c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], 
		*a_param, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size, 
		current->c_network->TC_scale_factor, current->c_network->iter * current->c_network->train.size);
	cuda_check_yolo_kernel("YOLO_deriv_error");
}


void cuda_YOLO_output_error(layer *current)
{
	yolo_param *a_param = (yolo_param*)current->activ_param;
	conv_param *c_param = (conv_param*)current->param;
	cu_blocks = ((size_t)c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] *
			current->c_network->batch_size + cu_threads - 1) / cu_threads;
	
	current->c_network->cu_inst.cu_YOLO_activ_fcts.output_error_fct<<< cu_blocks, cu_threads >>>
		((float*)current->c_network->output_error, current->output, current->c_network->target, current->c_network->output_dim, 
		c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2], c_param->nb_area[0], c_param->nb_area[1], c_param->nb_area[2], 
		*a_param, c_param->nb_area[0] * c_param->nb_area[1] * c_param->nb_area[2] * current->c_network->batch_size);
	cuda_check_yolo_kernel("YOLO_output_error");
}


void cuda_YOLO_activ_init(layer *current)
{
	float *temp_tab, *temp_tab2, **temp_tab3;
	
	size_t nb_area_flat;

	yolo_param* a_param = (yolo_param*)current->activ_param;
	
	nb_area_flat = ((conv_param*)current->param)->nb_area[0]
		* ((conv_param*)current->param)->nb_area[1]
		* ((conv_param*)current->param)->nb_area[2];
	
	switch(a_param->IoU_type)
	{
		case IOU:
		case ROTIOU:
			cudaMemcpyFromSymbol(&(a_param->c_IoU_fct), device_gpu_IoU_fct, sizeof(pointFunction_gpu_IoU));
			break;
			
		default:
		case GIOU:
			cudaMemcpyFromSymbol(&(a_param->c_IoU_fct), device_gpu_GIoU_fct, sizeof(pointFunction_gpu_IoU));
			break;
			
		case DIOU:
			cudaMemcpyFromSymbol(&(a_param->c_IoU_fct), device_gpu_DIoU_fct, sizeof(pointFunction_gpu_IoU));
			break;
		
		case DIOU2:
			cudaMemcpyFromSymbol(&(a_param->c_IoU_fct), device_gpu_DIoU2_fct, sizeof(pointFunction_gpu_IoU));
			break;
	}
	
	cuda_convert_table_FP32((void**)&(a_param->prior_size), a_param->nb_box * 3, 1);
	cuda_convert_table_FP32((void**)&(a_param->noobj_prob_prior), a_param->nb_box, 1);
	cuda_convert_table_FP32((void**)&(a_param->scale_tab), 6, 1);
	
	temp_tab = a_param->slopes_and_maxes_tab[0];
	cudaMalloc(&temp_tab2, 6 * 3 * sizeof(float));
	cudaMemcpy(temp_tab2, temp_tab, 6 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	for(int i = 0; i < 6; i++)
		a_param->slopes_and_maxes_tab[i] = &temp_tab2[i*3];
	temp_tab3 = a_param->slopes_and_maxes_tab;
	cudaMalloc(&(a_param->slopes_and_maxes_tab), 6 * sizeof(float*));
	cudaMemcpy(a_param->slopes_and_maxes_tab, temp_tab3, 6 * sizeof(float*), cudaMemcpyHostToDevice);
	
	cuda_convert_table_FP32((void**)&(a_param->param_ind_scale), a_param->nb_param, 1);
	cuda_convert_table_FP32((void**)&(a_param->IoU_limits), 8, 1);
	cuda_convert_table_int(&(a_param->fit_parts), 6, 1);
	
	cudaMalloc((void**)(&(a_param->block_state)), ((conv_param*)current->param)->nb_filters 
			* nb_area_flat * current->c_network->batch_size * sizeof(curandState_t));
	cu_blocks = ((((conv_param*)current->param)->nb_filters * current->c_network->batch_size 
		* (size_t)(nb_area_flat))  + cu_threads - 1) / cu_threads;
	init_block_state<<< cu_blocks, cu_threads>>>(time(NULL),(curandState_t*)(a_param->block_state), 
		((conv_param*)current->param)->nb_filters * nb_area_flat * current->c_network->batch_size);
	
	cuda_convert_table_int(&(a_param->cell_size), 3, 0);
	cuda_convert_table_FP32((void**)&(a_param->IoU_monitor),
		2 *a_param->nb_box * current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_int(&(a_param->target_cell_mask),
		a_param->max_nb_obj_per_image * current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_FP32((void**)&(a_param->IoU_table),
		a_param->max_nb_obj_per_image * a_param->nb_box 
		* current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_FP32((void**)&(a_param->dist_prior),
		a_param->max_nb_obj_per_image * a_param->nb_box 
		* current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_int(&(a_param->box_locked),
		a_param->nb_box * current->c_network->batch_size * nb_area_flat, 0);
	cuda_convert_table_FP32((void**)&(a_param->box_in_pix),
		6 * a_param->nb_box * current->c_network->batch_size * nb_area_flat, 0);
}


void cuda_free_yolo_activ_param(layer *current)
{
	yolo_param* a_param = (yolo_param*)current->activ_param;
	float **temp_tab;
	
	cudaFree(a_param->prior_size);
	cudaFree(a_param->noobj_prob_prior);
	cudaFree(a_param->scale_tab);
	
	temp_tab = (float**) malloc(6*sizeof(float*));
	cudaMemcpy(temp_tab, a_param->slopes_and_maxes_tab, 6 * sizeof(float*), cudaMemcpyDeviceToHost);
	cudaFree(temp_tab[0]);
	free(temp_tab);
	cudaFree(a_param->slopes_and_maxes_tab);
	
	cudaFree(a_param->param_ind_scale);
	cudaFree(a_param->IoU_limits);
	cudaFree(a_param->fit_parts);
	cudaFree(a_param->block_state);
	
	cudaFree(a_param->cell_size);
	cudaFree(a_param->IoU_monitor);
	cudaFree(a_param->target_cell_mask);
	cudaFree(a_param->IoU_table);
	cudaFree(a_param->dist_prior);
	cudaFree(a_param->box_locked);
	cudaFree(a_param->box_in_pix);
}


//#####################################################
//		 GENERAL FUNCTION ASSOCIATIONS
//#####################################################


void init_typed_cuda_activ(network* net)
{
	switch(net->cu_inst.use_cuda_TC)
	{
		default:
		case FP32C_FP32A:
		case TF32C_FP32A:
			typed_cuda_activ_fct_association_FP32(net);
			break;
			
		case FP16C_FP32A:
		case FP16C_FP16A:
			#if defined(GEN_VOLTA) || defined(GEN_AMPERE)
			typed_cuda_activ_fct_association_FP16(net);
			#else
			printf("ERROR: CIANNA not compiled with FP16 compute capability (GEN_VOLTA minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
		
		case BF16C_FP32A:
			#if defined(GEN_AMPERE)
			typed_cuda_activ_fct_association_BF16(net);
			#else
			printf("ERROR: CIANNA not compiled with BF16 compute capability (GEN_AMPERE minimum)\n");
			exit(EXIT_FAILURE);
			#endif
			break;
	}
}


void cuda_define_activation(layer *current)
{	
	switch(current->activation_type)
	{
		case RELU:
			current->activation = cuda_ReLU_activation;
			current->deriv_activation = cuda_ReLU_deriv;
			break;
		
		case LOGISTIC:
			current->activation = cuda_logistic_activation;
			current->deriv_activation = cuda_logistic_deriv;
			break;
			
		case SOFTMAX:
			current->activation = cuda_softmax_activation;
			current->deriv_activation = cuda_softmax_deriv;
			break;
			
		case YOLO:
			current->activation = cuda_YOLO_activation;
			current->deriv_activation = cuda_YOLO_deriv;
			cuda_YOLO_activ_init(current);
			break;
			
		case LINEAR:
		default:
			current->activation = cuda_linear_activation;
			current->deriv_activation = cuda_linear_deriv;
			break;
	}
}


void cuda_deriv_output_error(layer *current)
{
	switch(current->activation_type)
	{
		case RELU:
			cuda_ReLU_deriv_output_error(current);
			break;
		
		case LOGISTIC:
			cuda_logistic_deriv_output_error(current);
			break;
			
		case SOFTMAX:
			cuda_softmax_deriv_output_error(current);
			break;
			
		case YOLO:
			cuda_YOLO_deriv_output_error(current);
			break;
			
		case LINEAR:
		default:
			cuda_linear_deriv_output_error(current);
			break;
	
	}
}

void cuda_output_error_fct(layer* current)
{
	switch(current->activation_type)
	{
		case RELU:
			cuda_ReLU_output_error(current);
			break;
		
		case LOGISTIC:
			cuda_logistic_output_error(current);
			break;
			
		case SOFTMAX:
			cuda_softmax_output_error(current);
			break;
			
		case YOLO:
			cuda_YOLO_output_error(current);
			break;
			
		case LINEAR:
		default:
			cuda_linear_output_error(current);
			break;
	
	}
}




//#####################################################
