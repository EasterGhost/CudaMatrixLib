#pragma once
#ifndef KERNEL_FUNCTION_H
#define KERNEL_FUNCTION_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <sm_20_intrinsics.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>

using namespace std;

template <typename Type>
__global__ static void identity_matrix_kernel(Type* data, const int size);

template <typename Type>
__global__ static void ones_matrix_kernel(Type* data, const int total_elements);

template <typename Type>
__global__ static void random_matrix_kernel
(Type* data, const int total_elements, curandState* states);

__global__ static void setup_random_kernel
(curandState* state, size_t seed, const int size);

template <typename Type>
__global__ static void col_vec_broadcast_kernel
(const Type* src_vec, Type* res, const int size, const int cols);

template <typename Type>
__global__ static void row_vec_broadcast_kernel
(const Type* src_vec, Type* res, const int size, const int cols);

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_add_kernel
(const T1* src1, const T2* src2, T3* res, const int size);

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_subtract_kernel
(const T1* src1, const T2* src2, T3* res, const int size);

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_multiply_kernel
(const T1* src1, const T2* src2, T3* res, const int size);

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_divide_kernel
(const T1* src1, const T2* src2, T3* res, const int size);

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_add_kernel
(const T1* src, const T2 scalar, T3* res, const int size);

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_subtract_kernel
(const T1* src, const T2 scalar, T3* res, const int size);

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_multiply_kernel
(const T1* src, const T2 scalar, T3* res, const int size);

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_divide_kernel
(const T1* src, const T2 scalar, T3* res, const int size);

template <typename T1, typename T2, typename T3>
__global__ static void matrix_multiply_kernel
(const T1* src1, const T2* src2, T3* res, const int rows1, const int cols1, const int cols2);

#endif // !KERNEL_FUNCTION_H