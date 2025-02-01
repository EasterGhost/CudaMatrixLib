#pragma once
#include "kernel_function.cuh"

template <typename Type>
__global__ static void identity_matrix_kernel(Type* data, const unsigned int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		data[idx * size + idx] = (Type)1;
}

template <typename Type>
__global__ static void ones_matrix_kernel(Type* data, const unsigned int total_elements)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		data[idx] = (Type)1;
}

__global__ static void float_random_matrix_kernel
(float* data, const unsigned int total_elements, curandStatePhilox4_32_10_t* states)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		data[idx] = curand_uniform(&states[idx]);
}

__global__ static void float_qrandom_matrix_kernel
(float* data, curandStateScrambledSobol32_t* states, const unsigned int n, const unsigned int dimensions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		for (int d = 0; d < dimensions; d++)
			data[idx * dimensions + d] = curand_uniform(&states[idx * dimensions + d]);
}

__global__ static void double_random_matrix_kernel
(double* data, const unsigned int total_elements, curandStatePhilox4_32_10_t* states)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		data[idx] = curand_uniform_double(&states[idx]);
}

__global__ static void double_qrandom_matrix_kernel
(double* data, curandStateScrambledSobol64_t* states, const unsigned int n, const unsigned int dimensions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		for (int d = 0; d < dimensions; d++)
			data[idx * dimensions + d] = curand_uniform_double(&states[idx * dimensions + d]);
}

__global__ static void int_random_matrix_kernel
(int* data, const unsigned int total_elements, curandStatePhilox4_32_10_t* states)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		data[idx] = curand(&states[idx]);
}

__global__ static void int_qrandom_matrix_kernel
(int* data, curandStateScrambledSobol32_t* states, const unsigned int n, const unsigned int dimensions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		for (int d = 0; d < dimensions; d++)
			data[idx * dimensions + d] = curand(&states[idx * dimensions + d]);
}

__global__ static void setup_random_kernel
(curandStatePhilox4_32_10_t* states, size_t seed, const unsigned int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		curand_init(seed, idx, 0, states + idx);
}

__global__ static void setup_q32random_kernel
(curandStateScrambledSobol32_t* states, curandDirectionVectors32_t* dr_vec, const unsigned int n, const unsigned int dimensions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		for (int d = 0; d < dimensions; d++)
			curand_init(dr_vec[d], idx, 0, states + (idx * dimensions + d));
}

__global__ static void setup_q64random_kernel
(curandStateScrambledSobol64_t* states, curandDirectionVectors64_t* dr_vec, const unsigned int n, const unsigned int dimensions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		for (int d = 0; d < dimensions; d++)
			curand_init(dr_vec[d], idx, 0, states + (idx * dimensions + d));
}

template <typename Type>
__global__ static void col_vec_broadcast_kernel
(const Type* src_vec, Type* res, const int size, const int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		int row = idx / cols;
		res[idx] = src_vec[row];
	}
}

template <typename Type>
__global__ static void row_vec_broadcast_kernel
(const Type* src_vec, Type* res, const int size, const int cols)
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
(const T1* src1, const T2* src2, T3* res, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src1[idx] + src2[idx]);
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_subtract_kernel
(const T1* src1, const T2* src2, T3* res, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src1[idx] - src2[idx]);
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_multiply_kernel
(const T1* src1, const T2* src2, T3* res, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src1[idx] * src2[idx]);
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_divide_kernel
(const T1* src1, const T2* src2, T3* res, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src1[idx] / src2[idx]);
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_add_kernel
(const T1* src, const T2 scalar, T3* res, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src[idx] + scalar);
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_subtract_kernel
(const T1* src, const T2 scalar, T3* res, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src[idx] - scalar);
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_multiply_kernel
(const T1* src, const T2 scalar, T3* res, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src[idx] * scalar);
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_divide_kernel
(const T1* src, const T2 scalar, T3* res, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = (T3)(src[idx] / scalar);
}

template <typename T1, typename T2, typename T3>
__global__ static void matrix_multiply_kernel
(const T1* src1, const T2* src2, T3* res, const int rows1, const int cols1, const int cols2)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = blockIdx.z * blockDim.z + threadIdx.z;

	if (row < rows1 && col < cols2)
	{
		T3 sum = 0;
		for (int i = 0; i < cols1; i++)
			sum += src1[row * cols1 + i] * src2[i * cols2 + col];
		res[row * cols2 + col] = sum;
	}
}