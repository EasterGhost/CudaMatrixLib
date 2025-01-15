/**
* @file TemplateCudaMatrix.cuh
* @brief CUDA ������ͷ�ļ�
* @note ��֧�� float ����
* @note ʹ�� cuBLAS ʵ�־�������
* @date 2024-12-16
* @version 1.0
* @author Andrew Elizabeth
* @copyright MIT License
* @note ���ļ��еĴ����֧���� CUDA �����±��������
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
* @brief ��������
* @enum MatrixType
* @note ���ڱ�ʶ�����ض����͵ľ���
* @note Zero: ȫ�����
* @note Ones: ȫһ����
* @note Identity: ��λ����
* @note Random: �������
*/
typedef enum __device_builtin__ MatrixType {
	Zero,
	Ones,
	Identity,
	Random
}MatrixType;

/**
* @brief ���� CUDA �˺������߳̿��С
* @tparam T CUDA �˺���
* @param[in] func CUDA �˺���
* @return �߳̿��С
* @throw runtime_error �������ʧ��
* @note ʹ�� cudaOccupancyMaxPotentialBlockSize ���������߳̿��С
*/
template<class T>
static int autoSetBlockSize(T func);

/**
* @brief ���� CUDA �˺������߳̿��С
* @tparam T CUDA �˺���
* @param[in] func CUDA �˺���
* @param[in] rows ��������
* @param[in] cols ��������
* @return �߳̿��С
* @throw runtime_error �������ʧ��
* @note ʹ�� cudaOccupancyMaxPotentialBlockSize ���������߳̿��С
*/
template<class T>
static dim3 autoSetBlockSize2D(T func, int rows, int cols);

/**
* @brief CUDA �����Ⱦ�����
* @class cudaMatrix
*/
template <typename Type>
class CudaMatrix
{
private:
	int rows; /// ��������
	int cols; /// ��������
	Type* mat; /// ��������
	cublasHandle_t handle; /// cuBLAS ���
	cusolverDnHandle_t solver_handle; /// cuSOLVER ���
	cudaStream_t stream; /// CUDA ��
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
	* @brief ��ȡ��������
	* @return ��������
	*/
	int getRows() const;
	/**
	* @brief ��ȡ��������
	* @return ��������
	*/
	int getCols() const;
	/**
	* @brief ��ȡ��������
	* @return ��������
	*/
	Type* getData() const;
	/**
	* @brief ���þ�������
	* @param[in] data ��������
	*/
	void setData(const vector<Type>& data);
	/**
	* @brief ��ȡ����Ԫ��
	* @param[in] i ������
	* @param[in] j ������
	* @return ����Ԫ��
	*/
	Type get(const int i, const int j) const;
	/**
	* @brief ���þ���Ԫ��
	* @param[in] i ������
	* @param[in] j ������
	* @param[in] value Ԫ��ֵ
	*/
	void set(const int i, const int j, const Type value);

	Type* data() const;

	void print();
};
#endif // !TEMPLATE_CUDA_MATRIX_H