/**
* @file TemplateCudaMatrix.cuh
* @brief CUDA ������ͷ�ļ�
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
#define pi 3.1415926535897932
#endif // !pi

#ifndef O
#define O CudaMatrix(1)
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
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <sm_60_atomic_functions.h>
#include <sm_61_intrinsics.h>
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
#include <stdexcept>

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
typedef enum __device_builtin__ MatrixType : uint8_t
{
	Identity = 255,
	Zero = 0,
	Ones,
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
	unsigned int rows; /// ��������
	unsigned int cols; /// ��������
	Type* mat; /// ��������
	cublasHandle_t handle; /// cuBLAS ���
	cusolverDnHandle_t solver_handle; /// cuSOLVER ���
	//cudaStream_t stream; /// CUDA ��
public:
	/**
	 * @brief Ĭ�Ϲ��캯��
	 */
	CudaMatrix();
	/**
	 * @brief ���������캯����0��ʼ����
	 * @param rows ��������
	 * @param cols ��������
	 */
	CudaMatrix(const unsigned int rows, const unsigned int cols);
	CudaMatrix(const unsigned int size);
	/**
	 * @brief ���������캯����ָ�����ͣ�
	 * @param rows ��������
	 * @param cols ��������
	 * @param type ��������
	 */
	CudaMatrix(const unsigned int rows, const unsigned int cols, const MatrixType type);
	CudaMatrix(const unsigned int size, const MatrixType type);
	/**
	 * @brief ���ݹ��캯��
	 * @param rows ��������
	 * @param cols ��������
	 * @param data ��������
	 */
	CudaMatrix(const unsigned int rows, const unsigned int cols, const Type* src);
	CudaMatrix(const unsigned int rows, const unsigned int cols, const vector<Type>& src);
	CudaMatrix(const unsigned int size, const Type* src);
	CudaMatrix(const unsigned int size, const vector<Type>& src);
	/**
	 * @brief �������캯��
	 * @param other ��һ��cuda����
	 */
	CudaMatrix(const CudaMatrix<Type>& other);
	~CudaMatrix();
	/**
	* @brief ��ȡ��������
	* @return ��������
	*/
	unsigned int getRows() const;
	/**
	* @brief ��ȡ��������
	* @return ��������
	*/
	unsigned int getCols() const;
	/**
	* @brief ��ȡ��������
	* @return ��������
	*/
	void getData(Type* dst) const;
	/**
	* @brief ���þ�������
	* @param[in] data ��������
	*/
	void setData(const vector<Type>& src);
	/**
	* @brief ��ȡ����Ԫ��
	* @param[in] i ������
	* @param[in] j ������
	* @return ����Ԫ��
	*/
	Type get(const unsigned int row, const unsigned int col) const;
	/**
	* @brief ���þ���Ԫ��
	* @param[in] i ������
	* @param[in] j ������
	* @param[in] value Ԫ��ֵ
	*/
	void set(const unsigned int row, const unsigned int col, const Type value);

	Type* data() const;

	void print();

	template<typename T>
	void add(const CudaMatrix<T>& other);
};
#endif // !TEMPLATE_CUDA_MATRIX_H