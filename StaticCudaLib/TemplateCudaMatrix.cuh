/**
* @file TemplateCudaMatrix.cuh
* @brief CUDA 矩阵类头文件
* @note 仅支持 float 类型
* @note 使用 cuBLAS 实现矩阵运算
* @date 2024-12-16
* @version 1.0
* @author Andrew Elizabeth
* @copyright MIT License
* @note 本文件中的代码仅支持在 CUDA 环境下编译和运行
* @example examples/cuda_matrix_example.cpp
*/

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

/**
* @brief CUDA 行优先矩阵类
* @class cudaMatrix
*/
template <typename Type>
class CudaMatrix
{
private:
	int rows; /// 矩阵行数
	int cols; /// 矩阵列数
	Type* mat; /// 矩阵数据
	cublasHandle_t handle; /// cuBLAS 句柄
	cusolverDnHandle_t solver_handle; /// cuSOLVER 句柄
	cudaStream_t stream; /// CUDA 流
public:
	CudaMatrix();
	CudaMatrix(int rows, int cols);
	CudaMatrix(int rows, int cols, MatrixType type);
	CudaMatrix(int size);
	CudaMatrix(int size, MatrixType type);
	CudaMatrix(int rows, int cols, Type* data);
	CudaMatrix(int rows, int cols, vector<Type> data);
	CudaMatrix(int size, Type* data);
	CudaMatrix(int size, vector<Type> data);
	CudaMatrix(const CudaMatrix<Type>& others);
	~CudaMatrix();
	/**
	* @brief 获取矩阵行数
	* @return 矩阵行数
	*/
	int getRows() const;
	/**
	* @brief 获取矩阵列数
	* @return 矩阵列数
	*/
	int getCols() const;
	/**
	* @brief 获取矩阵数据
	* @return 矩阵数据
	*/
	Type* getData() const;
	/**
	* @brief 设置矩阵数据
	* @param[in] data 矩阵数据
	*/
	void setData(const vector<Type>& data);
	/**
	* @brief 获取矩阵元素
	* @param[in] i 行索引
	* @param[in] j 列索引
	* @return 矩阵元素
	*/
	Type get(const int i, const int j) const;
	/**
	* @brief 设置矩阵元素
	* @param[in] i 行索引
	* @param[in] j 列索引
	* @param[in] value 元素值
	*/
	void set(const int i, const int j, const Type value);

	Type* data() const;

	void print();
};
#endif // !TEMPLATE_CUDA_MATRIX_H