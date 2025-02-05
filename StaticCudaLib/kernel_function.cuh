#pragma once
#ifndef KERNEL_FUNCTION_H
#define KERNEL_FUNCTION_H
#include <crt/host_defines.h>
#include <curand_kernel.h>
#include <curand_mtgp32_kernel.h>
#include <curand_philox4x32_x.h>
#include <curand_uniform.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace std;

template <typename T1, typename T2>
__global__ static void convert_kernel(const T1* src, T2* res, const size_t size);

template <typename Type>
__global__ static void assign_kernel(Type* data, const Type value, const size_t size);

template <typename Type>
__global__ static void identity_matrix_kernel(Type* data, const size_t size);

template <typename Type>
__global__ static void ones_matrix_kernel(Type* data, const size_t total_elements);

__global__ static void float_random_matrix_kernel
(float* data, const size_t total_elements, curandStatePhilox4_32_10_t* states);

__global__ static void float_qrandom_matrix_kernel
(float* data, curandStateScrambledSobol32_t* states, const uint32_t n, const uint32_t dimensions);

__global__ static void double_random_matrix_kernel
(double* data, const size_t total_elements, curandStatePhilox4_32_10_t* states);

__global__ static void double_qrandom_matrix_kernel
(double* data, curandStateScrambledSobol64_t* states, const uint32_t n, const uint32_t dimensions);

__global__ static void int_random_matrix_kernel
(int* data, const size_t total_elements, curandStatePhilox4_32_10_t* states);

__global__ static void int_qrandom_matrix_kernel
(int* data, curandStateScrambledSobol32_t* states, const uint32_t n, const uint32_t dimensions);

__global__ static void setup_random_kernel
(curandStatePhilox4_32_10_t* state, size_t seed, const size_t size);

__global__ static void setup_q32random_kernel
(curandStateScrambledSobol32_t* states, curandDirectionVectors32_t* dr_vec, const uint32_t size, const uint32_t dimensions);

__global__ static void setup_q64random_kernel
(curandStateScrambledSobol64_t* states, curandDirectionVectors64_t* dr_vec, const uint32_t n, const uint32_t dimensions);

template <typename Type>
__global__ static void matrix_transpose_kernel
(const Type* src, Type* res, const uint32_t rows, const uint32_t cols);

template <typename Type>
__global__ static void elementwise_equal_kernel
(const Type* src1, const Type* src2, bool* res, const uint32_t size);

template <typename Type> __global__ static void col_vec_broadcast_kernel
(const Type* src_vec, Type* res, const uint32_t size, const uint32_t cols);

template <typename Type> __global__ static void row_vec_broadcast_kernel
(const Type* src_vec, Type* res, const uint32_t size, const uint32_t cols);

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_add_kernel
(const T1* src1, const T2* src2, T3* res, const uint32_t size);

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_subtract_kernel
(const T1* src1, const T2* src2, T3* res, const uint32_t size);

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_multiply_kernel
(const T1* src1, const T2* src2, T3* res, const uint32_t size);

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_divide_kernel
(const T1* src1, const T2* src2, T3* res, const uint32_t size);

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_add_kernel
(const T1* src, const T2 scalar, T3* res, const uint32_t size);

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_subtract_kernel
(const T1* src, const T2 scalar, T3* res, const uint32_t size);

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_multiply_kernel
(const T1* src, const T2 scalar, T3* res, const uint32_t size);

template <typename T1, typename T2, typename T3>
__global__ static void matrix_scalar_divide_kernel
(const T1* src, const T2 scalar, T3* res, const uint32_t size);

template <typename T1, typename T2, typename T3>
__global__ static void matrix_multiply_kernel
(const T1* src1, const T2* src2, T3* res, const uint32_t rows1, const uint32_t cols1, const uint32_t cols2);

template <typename T> __global__ static void reshape_kernel
(const T* src, T* res, const uint32_t rows_old, const uint32_t cols_old, const uint32_t rows_new, const uint32_t cols_new);
#endif // !KERNEL_FUNCTION_H