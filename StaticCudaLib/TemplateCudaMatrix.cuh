#pragma once
#ifndef NUM_THREADS
#define NUM_THREADS 96
#endif // !NUM_THREADS

#ifndef pi
#define pi 3.1415926535897932384626433832795
#endif // !pi

#ifndef O
#define O cudaMatrix(1)
#endif // !O

#ifndef FORCE_SAFE_SIZE
#define FORCE_SAFE_SIZE true
#endif // !FORCE_SAFE_SIZE

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
	do { \
		cudaError_t err = call; \
		if (err != cudaSuccess) { \
			throw runtime_error("CUDA error: " + string(cudaGetErrorString(err))); \
		} \
	} while (0)
#endif // !CUDA_CHECK(call)

#ifndef TEMPLATE_CUDA_MATRIX_H
#define TEMPLATE_CUDA_MATRIX_H
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <sm_20_intrinsics.h>
//#include <device_double_functions.h>
//#include <device_atomic_functions.hpp>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>
#include <cstdarg>
#include <ctime>
#include <cmath>
#include <thread>
#include <limits>
#include <mutex>

using namespace std;

/**
* @brief 矩阵类型
* @enum MatrixType
* @note 用于标识生成特定类型的矩阵
* @note Zero: 全零矩阵
* @note Ones: 全一矩阵
* @note Identity: 单位矩阵
* @note Random: 随机矩阵
*/
typedef enum __device_builtin__ MatrixType {
	Zero,
	Ones,
	Identity,
	Random
}MatrixType;

/**
* @brief 设置 CUDA 核函数的线程块大小
* @tparam T CUDA 核函数
* @param[in] func CUDA 核函数
* @return 线程块大小
* @throw runtime_error 如果设置失败
* @note 使用 cudaOccupancyMaxPotentialBlockSize 函数设置线程块大小
*/
template<class T>
static int autoSetBlockSize(T func);

/**
* @brief 设置 CUDA 核函数的线程块大小
* @tparam T CUDA 核函数
* @param[in] func CUDA 核函数
* @param[in] rows 矩阵行数
* @param[in] cols 矩阵列数
* @return 线程块大小
* @throw runtime_error 如果设置失败
* @note 使用 cudaOccupancyMaxPotentialBlockSize 函数设置线程块大小
*/
template<class T>
static dim3 autoSetBlockSize2D(T func, int rows, int cols);

template<typename Type>
class cudaMatrix {
private:
	int rows;
	int cols;
	Type* data;
public:
	cudaMatrix<Type>();
	cudaMatrix<Type>(int rows, int cols);
	cudaMatrix<Type>(int rows, int cols, MatrixType type);
	cudaMatrix<Type>(int size);
	cudaMatrix<Type>(int size, MatrixType type);
	cudaMatrix<Type>(int rows, int cols, Type* data);
	cudaMatrix<Type>(const cudaMatrix<Type>& other);
};

#endif // !TEMPLATE_CUDA_MATRIX_H