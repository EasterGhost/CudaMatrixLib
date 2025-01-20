#include "kernel_function.cuh"

template <typename Type>
__global__ static void identity_matrix_kernel(Type* data, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		data[idx * size + idx] = (Type)1;
}

template <typename Type>
__global__ static void ones_matrix_kernel(Type* data, const int total_elements)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		data[idx] = (Type)1;
}

template <typename Type>
__global__ static void random_matrix_kernel
(Type* data, const int total_elements, curandState* states)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		if constexpr (is_floating_point<Type>::value)
			data[idx] = curand_uniform(states + idx);
		else if constexpr (is_same<Type, double>::value)
			data[idx] = curand_uniform_double(states + idx);
		else
			data[idx] = (Type)curand(states + idx);
}

__global__ static void setup_random_kernel
(curandState* state, size_t seed, const int size)
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