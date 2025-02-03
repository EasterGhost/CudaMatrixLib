/**
* @file TemplateCudaMatrix.cuh
* @brief CUDA 矩阵类头文件
* @note 使用 cuBLAS 实现矩阵运算
* @date 2025-2-16
* @version 2.0
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
#define pi 3.14159265358979323846
#endif // !pi

#ifndef O
#define O CudaMatrix(1)
#endif // !O

#ifndef FORCE_SAFE_SIZE
#define FORCE_SAFE_SIZE true
#endif // !FORCE_SAFE_SIZE

#ifndef IS_SAFE_DATA
#define IS_SAFE_DATA true
#endif // !IS_SAFE_DATA

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
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <ctime>
#include <cstdarg>
#include <curand.h>
#include <curand_philox4x32_x.h>
#include <driver_types.h>
#include <curand_kernel.h>
#include <curand_uniform.h>
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
typedef enum __device_builtin__ MatrixType : uint8_t
{
	Identity = 255,
	Zero = 0,
	Ones,
	Random,
	QuasiRandom
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
public:
	/**
	* @brief basic typedefs
	*/
	using value_type = Type;
	using size_type = size_t;
	using difference_type = ptrdiff_t;
	/**
	* @brief pointer typedefs
	*/
	using pointer = Type*;
	using const_pointer = const Type*;
	/**
	* @brief reference typedefs
	*/
	using reference = Type&;
	using const_reference = const Type&;
	/**
	 * @brief 
	 */
	using iterator = pointer;
	/**
	 * @brief 默认构造函数
	 */
	CudaMatrix();
	/**
	 * @brief 参数化构造函数（0初始化）
	 * @param rows 矩阵行数
	 * @param cols 矩阵列数
	 */
	CudaMatrix(const uint32_t rows, const uint32_t cols);
	explicit CudaMatrix(const uint32_t size);
	/**
	 * @brief 参数化构造函数（指定类型）
	 * @param rows 矩阵行数
	 * @param cols 矩阵列数
	 * @param type 矩阵类型
	 */
	CudaMatrix(const uint32_t rows, const uint32_t cols, const MatrixType type);
	CudaMatrix(const uint32_t size, const MatrixType type);
	/**
	 * @brief 数据构造函数
	 * @param rows 矩阵行数
	 * @param cols 矩阵列数
	 * @param data 矩阵数据
	 */
	CudaMatrix(const uint32_t rows, const uint32_t cols, const pointer src);
	CudaMatrix(const uint32_t rows, const uint32_t cols, const vector<value_type>& src);
	CudaMatrix(const uint32_t size, const pointer src);
	CudaMatrix(const uint32_t size, const vector<value_type>& src);

	/**
	 * @brief 移动构造函数
	 * @param other 另一个cuda矩阵
	 */
	CudaMatrix(CudaMatrix&& other) noexcept;
	/**
	 * @brief 拷贝构造函数
	 * @param other 另一个cuda矩阵
	 */
	CudaMatrix(const CudaMatrix<value_type>& other);
	/**
	 * @brief 拷贝赋值运算符
	 * @param other 另一个cuda矩阵
	 * @return 当前cuda矩阵
	 */
	CudaMatrix& operator=(const CudaMatrix<value_type>& other);
	/**
	 * @brief 移动赋值运算符
	 * @param other 另一个cuda矩阵
	 * @return 当前cuda矩阵
	 */
	CudaMatrix& operator=(CudaMatrix&& other) noexcept;
	/**
	* @brief 析构函数
	*/
	~CudaMatrix();

	/**
	* @brief 重载运算符
	*/
	bool operator==(const CudaMatrix<value_type>& other) const;
	bool operator!=(const CudaMatrix<value_type>& other) const;
	size_type size() const noexcept;

	bool empty() const noexcept;

	constexpr size_type max_size() const noexcept;

	/**
	* @brief 获取矩阵行数
	* @return 矩阵行数
	*/
	unsigned int getRows() const;
	/**
	* @brief 获取矩阵列数
	* @return 矩阵列数
	*/
	unsigned int getCols() const;
	/**
	* @brief 获取矩阵数据
	* @return 矩阵数据
	*/
	void getData(pointer dst) const;
	void getData(vector<value_type>& dst) const;
	void getData(vector<value_type>& dst, bool is_safesize) const;
	/**
	* @brief 设置矩阵数据
	* @param[in] src 矩阵数据
	*/
	void setData(const pointer src);
	void setData(const vector<value_type>& src);
	/**
	* @brief 获取矩阵元素
	* @param[in] i 行索引
	* @param[in] j 列索引
	* @return 矩阵元素
	*/
	value_type get(const uint32_t row, const uint32_t col) const;
	/**
	* @brief 设置矩阵元素
	* @param[in] i 行索引
	* @param[in] j 列索引
	* @param[in] value 元素值
	*/
	void set(const uint32_t row, const uint32_t col, const value_type value);

	pointer data() noexcept;

	const_pointer data() const noexcept;

	void print();
	void printMatrix();

	void resize(const uint32_t rows, const uint32_t cols) noexcept;

	void resize(const uint32_t size) noexcept;

	void updateDimensions(const uint32_t rows, const uint32_t cols);

	void updateDimensions(const uint32_t size);

	void reshape(const uint32_t rows, const uint32_t cols);

	void reshape(const uint32_t size);

	template<typename T>
	void add(const CudaMatrix<T>& other);

private:
	unsigned int rows, cols; /// 矩阵行数和列数
	pointer mat; /// 矩阵数据
	cublasHandle_t handle; /// cuBLAS 句柄
	cusolverDnHandle_t solver_handle; /// cuSOLVER 句柄
	//cudaStream_t stream; /// CUDA 流

};
#endif // !TEMPLATE_CUDA_MATRIX_H