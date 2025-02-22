/*****************************************************************//**
 * \file   kernel_function.cu
 * \brief  CUDA kernel functions for matrix operations
 * \author AndrewElizabeth
 * \date   February 2025
 *********************************************************************/
#pragma once
#include "kernel_function.cuh"

template<typename T1, typename T2>
__global__ static void convert_kernel(const T1* src, T2* res, const size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = static_cast<T2>(src[idx]);
}

template<typename Type>
__global__ static void assign_kernel(Type* data, const Type value, const size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		data[idx] = value;
}

template <typename Type>
__global__ static void identity_matrix_kernel(Type* data, const size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		data[idx * size + idx] = (Type)1;
}

template <typename Type>
__global__ static void ones_matrix_kernel(Type* data, const size_t total_elements)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		data[idx] = (Type)1;
}

__global__ static void float_random_matrix_kernel
(float* data, const size_t total_elements, curandStatePhilox4_32_10_t* states)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		data[idx] = curand_uniform(&states[idx]);
}

__global__ static void float_qrandom_matrix_kernel
(float* data, curandStateScrambledSobol32_t* states, const uint32_t n, const uint32_t dimensions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		for (int d = 0; d < dimensions; d++)
			data[idx * dimensions + d] = curand_uniform(&states[idx * dimensions + d]);
}

__global__ static void double_random_matrix_kernel
(double* data, const size_t total_elements, curandStatePhilox4_32_10_t* states)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		data[idx] = curand_uniform_double(&states[idx]);
}

__global__ static void double_qrandom_matrix_kernel
(double* data, curandStateScrambledSobol64_t* states, const uint32_t n, const uint32_t dimensions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		for (int d = 0; d < dimensions; d++)
			data[idx * dimensions + d] = curand_uniform_double(&states[idx * dimensions + d]);
}

__global__ static void int_random_matrix_kernel
(int* data, const size_t total_elements, curandStatePhilox4_32_10_t* states)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		data[idx] = curand(&states[idx]);
}

__global__ static void int_qrandom_matrix_kernel
(int* data, curandStateScrambledSobol32_t* states, const uint32_t n, const uint32_t dimensions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		for (int d = 0; d < dimensions; d++)
			data[idx * dimensions + d] = curand(&states[idx * dimensions + d]);
}

__global__ static void setup_random_kernel
(curandStatePhilox4_32_10_t* states, size_t seed, const size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		curand_init(seed, idx, 0, states + idx);
}

__global__ static void setup_q32random_kernel
(curandStateScrambledSobol32_t* states, curandDirectionVectors32_t* dr_vec, const uint32_t n, const uint32_t dimensions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		for (int d = 0; d < dimensions; d++)
			curand_init(dr_vec[d], idx, 0, states + (idx * dimensions + d));
}

__global__ static void setup_q64random_kernel
(curandStateScrambledSobol64_t* states, curandDirectionVectors64_t* dr_vec, const uint32_t n, const uint32_t dimensions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		for (int d = 0; d < dimensions; d++)
			curand_init(dr_vec[d], idx, 0, states + (idx * dimensions + d));
}

template<typename Type> __global__ static void matrix_transpose_kernel
(const Type* src, Type* res, const uint32_t rows, const uint32_t cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < rows && idy < cols)
		res[idy * rows + idx] = src[idx * cols + idy];
}

template <typename Type> __global__ void elementwise_equal_kernel
(const Type* src1, const Type* src2, char* res, const size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (src1[idx] == src2[idx]);
}

template <typename Type> __global__ static void col_vec_broadcast_kernel
(const Type* src_vec, Type* res, const uint32_t size, const uint32_t cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		int row = idx / cols;
		res[idx] = src_vec[row];
	}
}

template <typename Type> __global__ static void row_vec_broadcast_kernel
(const Type* src_vec, Type* res, const uint32_t size, const uint32_t cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		int col = idx % cols;
		res[idx] = src_vec[col];
	}
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_add_kernel
(const T1* src1, const T2* src2, T3* res, const uint32_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src1[idx] + src2[idx]);
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_subtract_kernel
(const T1* src1, const T2* src2, T3* res, const uint32_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)src1[idx] - (T3)src2[idx];
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_multiply_kernel
(const T1* src1, const T2* src2, T3* res, const uint32_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src1[idx] * src2[idx]);
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_divide_kernel
(const T1* src1, const T2* src2, T3* res, const uint32_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src1[idx] / src2[idx]);
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_add_kernel
(const T1* src, const T2 scalar, T3* res, const uint32_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src[idx] + scalar);
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_subtract_kernel
(const T1* src, const T2 scalar, T3* res, const uint32_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src[idx] - scalar);
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_multiply_kernel
(const T1* src, const T2 scalar, T3* res, const uint32_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src[idx] * scalar);
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_divide_kernel
(const T1* src, const T2 scalar, T3* res, const uint32_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src[idx] / scalar);
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_multiply_kernel
(const T1* src1, const T2* src2, T3* res, const uint32_t rows1, const uint32_t cols1, const uint32_t cols2)
{
	const uint32_t row = blockIdx.x;
	const uint32_t col = blockIdx.y;
	extern __shared__ double shared_data[];
	double partial_sum = 0;
	const uint32_t tid = threadIdx.x;
	const uint32_t blockSize = blockDim.x;
	for (uint32_t i = tid; i < cols1; i += blockSize)
		partial_sum = (double)src1[row * cols1 + i] * (double)src2[i * cols2 + col] + partial_sum;
	shared_data[tid] = partial_sum;
	__syncthreads();
	for (uint32_t stride = blockSize / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
			shared_data[tid] += shared_data[tid + stride];
		__syncthreads();
	}
	if (tid == 0)
		res[row * cols2 + col] = (T3)shared_data[0];
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_multiply_kernel2
(const T1* src1, const T2* src2, T3* res, const uint32_t rows1, const uint32_t cols1, const uint32_t cols2)
{
	const uint32_t row = blockIdx.x;
	const uint32_t col = blockIdx.y;
	extern __shared__ T3 shared_data2[];
	T3 partial_sum = 0;
	const uint32_t tid = threadIdx.x;
	const uint32_t blockSize = blockDim.x;
	for (uint32_t i = tid; i < cols1; i += blockSize)
		partial_sum += (T3)src1[row * cols1 + i] * (T3)src2[i * cols2 + col];
	shared_data2[tid] = partial_sum;
	__syncthreads();
	for (uint32_t stride = blockSize / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
			shared_data2[tid] += shared_data2[tid + stride];
		__syncthreads();
	}
	if (tid == 0)
		res[row * cols2 + col] = (T3)shared_data2[0];
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_multiply_kernel3
(const T1* src1, const T2* src2, T3* res, const uint32_t rows1, const uint32_t cols1, const uint32_t cols2)
{
	const uint32_t row = blockIdx.x;
	const uint32_t col = blockIdx.y;
	extern __shared__ double shared_data[];
	double partial_sum = 0;
	const uint32_t tid = threadIdx.x;
	const uint32_t blockSize = blockDim.x;
	for (uint32_t i = tid; i < cols1; i += blockSize)
		partial_sum = __fma_rn((double)src1[row * cols1 + i], (double)src2[i * cols2 + col], partial_sum);
	shared_data[tid] = partial_sum;
	__syncthreads();
	for (uint32_t stride = blockSize / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
			shared_data[tid] += shared_data[tid + stride];
		__syncthreads();
	}
	if (tid == 0)
		res[row * cols2 + col] = (T3)shared_data[0];
}

template <typename T> __global__ static void reshape_kernel
(const T* src, T* res, const uint32_t rows_old, const uint32_t cols_old,const uint32_t rows_new, const uint32_t cols_new)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idx < rows_new && idy < cols_new && idx < rows_old && idy < cols_old)
		res[idx * cols_new + idy] = src[idx * cols_old + idy];
}
