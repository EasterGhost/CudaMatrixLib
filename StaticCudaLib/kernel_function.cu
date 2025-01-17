#include "kernel_function.cuh"
template <typename Type>
__global__ static void identity_matrix_kernel(Type* data, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		data[idx * size + idx] = 1;
}

template <typename Type>
__global__ static void ones_matrix_kernel(Type* data, const int total_elements)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		data[idx] = 1;
}

template <typename Type>
__global__ static void random_matrix_kernel
(Type* data, const int total_elements, curandState* states)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
	{
		curandState localState = states[idx];
		if constexpr (std::is_floating_point<Type>::value)
			data[idx] = curand_uniform(&localState);

		else if constexpr (is_same<Type, double>::value)
			data[idx] = curand_uniform_double(&localState);
		else
			data[idx] = curand(&localState);
	}
}

__global__ static void setup_random_kernel(curandState* state, size_t seed, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		curand_init(seed, idx, 0, &state[idx]);
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
		res[idx] = src1[idx] + src2[idx];
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_subtract_kernel
(const T1* src1, const T2* src2, T3* res, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = src1[idx] - src2[idx];
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_multiply_kernel
(const T1* src1, const T2* src2, T3* res, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = src1[idx] * src2[idx];
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_divide_kernel
(const T1* src1, const T2* src2, T3* res, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = src1[idx] / src2[idx];
}