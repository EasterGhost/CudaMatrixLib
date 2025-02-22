﻿/*****************************************************************//**
* \file   cuda_matrix.cuh
 * \brief  Implementation of a CUDA-based matrix template class that provides
 *         high-performance matrix operations and computations.
 *
 * \details Key features:
 *          - Supports various matrix initialization types (zero, identity, random)
 *          - Provides basic matrix operations
 *          - Implements efficient CUDA memory management
 *          - Integrates cuBLAS and cuSOLVER for accelerated computations
 *          - Supports safe element access and modification
 *          - Implements row-major matrix storage
 *          - Template-based implementation for different data types
 *          - RAII-compliant resource management
 *          - STL-compatible iterators
 *          - Thread-safe operations
 *
 * This class encapsulates CUDA kernel functions and matrix operations,
 * providing a high-level interface for GPU-accelerated matrix computations.
 *
 * \author AndrewElizabeth (easterghost1@gmail.com)
 * \date   February 2025
 *
 * \requirements
 *     - CUDA Runtime API
 *     - cuBLAS library
 *     - cuSOLVER library
 *	   - cuRAND library
 *
 * \note Thread safety is ensured through CUDA synchronization primitives
 * \warning Requires CUDA-capable hardware and proper CUDA environment setup
 *********************************************************************/

#pragma once

#ifndef TIME_USED
#define TIME_USED
#include <ctime>
extern clock_t time_used_init = 0;
extern clock_t time_used_gen_init = 0;
extern clock_t time_used_gen = 0;
extern clock_t time_used_switch_type = 0;
extern clock_t time_used_setblock = 0;
extern clock_t time_used_end = 0;
#endif // !tTIME_USED

#ifndef O
#define O cumatrix(1)
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
#include <cstdarg>
#include <curand.h>
#include <curand_philox4x32_x.h>
#include <driver_types.h>
#include <curand_kernel.h>
#include <curand_uniform.h>
#include "kernel_function.cu"
#include "kernel_function.cuh"
using namespace std;

typedef struct __device_builtin__ coord_t
{
	uint32_t x;
	uint32_t y;
}coord_t;

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
* @brief CUDA 内存分配器
* @tparam T 数据类型
*/
template <typename T>
class cuda_allocator
{
public:
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using size_type = size_t;
	using difference_type = ptrdiff_t;

	// 重绑定类型
	template <typename U>
	struct rebind { using other = cuda_allocator<U>; };

	cuda_allocator() = default;
	template <typename U>
	cuda_allocator(const cuda_allocator<U>&) noexcept {}

	// 分配 GPU 内存
	pointer allocate(size_type n)
	{
		pointer p;
		CUDA_CHECK(cudaMalloc((void**)&p, n * sizeof(T)));
		return p;
	}

	// 释放 GPU 内存
	void deallocate(pointer p, size_type) noexcept { CUDA_CHECK(cudaFree(p)); }

	// 在 GPU 上构造对象
	template<typename U, typename... Args>
	void construct(U* p, Args&&... args)
	{
		// 在主机上创建临时对象
		U temp(std::forward<Args>(args)...);
		CUDA_CHECK(cudaMemcpy(p, &temp, sizeof(U), cudaMemcpyHostToDevice));
	}

	// 销毁 GPU 上的对象
	template<typename U>
	void destroy(U*) noexcept {} // do nothing
};

/**
* @brief CUDA 行优先矩阵类
* @class cumatrix
*/
template <typename Type>
class cumatrix
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
	 * @brief iterator typedefs
	 */
	using iterator = pointer;
	using const_iterator = const_pointer;
	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;
	using allocator_type = cuda_allocator<Type>;
	/**
	 * @brief 默认构造函数
	 */
	cumatrix();
	/**
	 * @brief 参数化构造函数（0初始化）
	 * @param rows 矩阵行数
	 * @param cols 矩阵列数
	 */
	cumatrix(const uint32_t rows, const uint32_t cols);
	explicit cumatrix(const uint32_t size);
	/**
	 * @brief 参数化构造函数（指定类型）
	 * @param rows 矩阵行数
	 * @param cols 矩阵列数
	 * @param type 矩阵类型
	 */
	cumatrix(const uint32_t rows, const uint32_t cols, const MatrixType type);
	cumatrix(const uint32_t size, const MatrixType type);
	/**
	 * @brief 数据构造函数
	 * @param rows 矩阵行数
	 * @param cols 矩阵列数
	 * @param data 矩阵数据
	 */
	cumatrix(const uint32_t rows, const uint32_t cols, const pointer src);
	cumatrix(const uint32_t rows, const uint32_t cols, const vector<value_type>& src);
	cumatrix(const uint32_t size, const pointer src);
	cumatrix(const uint32_t size, const vector<value_type>& src);

	/**
	 * @brief 移动构造函数
	 * @param other 另一个cuda矩阵
	 */
	cumatrix(cumatrix&& other) noexcept;
	/**
	 * @brief 拷贝构造函数
	 * @param other 另一个cuda矩阵
	 */
	cumatrix(const cumatrix<value_type>& other);
	/**
	 * @brief 拷贝赋值运算符
	 * @param other 另一个cuda矩阵
	 * @return 当前cuda矩阵
	 */
	cumatrix& operator=(const cumatrix<value_type>& other);
	/**
	 * @brief 移动赋值运算符
	 * @param other 另一个cuda矩阵
	 * @return 当前cuda矩阵
	 */
	cumatrix& operator=(cumatrix&& other) noexcept;
	/**
	* @brief 析构函数
	*/
	~cumatrix();

	/**
	* @brief 重载运算符
	*/
	bool operator==(const cumatrix<value_type>& other) const;
	bool operator!=(const cumatrix<value_type>& other) const;

	bool empty() const noexcept;

	constexpr size_type max_size() const noexcept;

	reference front();
	const_reference front() const;
	reference back();
	const_reference back() const;
	value_type operator[](const coord_t coord) const;
	value_type operator[](const uint32_t index) const;
	void clear() noexcept;
	void swap(cumatrix<value_type>& other) noexcept;
	void assign(const cumatrix<value_type>& other);
	void assign(cumatrix<value_type>&& other) noexcept;
	void assign(const uint32_t rows, const uint32_t cols, const_reference val);
	void assign(const uint32_t size, const_reference val);
	void assign(const initializer_list<value_type>& il);
	void insert(const uint32_t rows, const uint32_t cols, const_reference val);
	value_type at(const uint32_t rows, const uint32_t cols) const;
	size_type capacity() const noexcept;
	size_type size() const noexcept;
	void reserve(const size_type new_cap);
	/**
	 * @brief 迭代器
	 * @note 提供迭代器支持
	 */
	iterator begin() noexcept { return mat; }
	const_iterator begin() const noexcept { return mat; }
	const_iterator cbegin() const noexcept { return mat; }
	iterator end() noexcept { return mat + rows * cols; }
	const_iterator end() const noexcept { return mat + rows * cols; }
	const_iterator cend() const noexcept { return mat + rows * cols; }
	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
	const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
	const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
	reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
	const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
	const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

	template <typename T>
	operator cumatrix<T>() const
	{
		cumatrix<T> result(rows, cols);
		int blockSize = autoSetBlockSize(convert_kernel<Type, T>);
		int gridSize = (rows * cols + blockSize - 1) / blockSize;
		convert_kernel<Type, T> << <gridSize, blockSize >> > (this->mat, result.data(), rows * cols);
		cudaDeviceSynchronize();
		return result;
	}
	uint32_t rowcount() const;
	uint32_t colcount() const;
	void get_data(pointer dst) const;
	void get_data(vector<value_type>& dst) const;
	void get_data(vector<value_type>& dst, bool is_safesize) const;
	void set_data(const pointer src);
	void set_data(const vector<value_type>& src);
	value_type get(const uint32_t row, const uint32_t col) const;
	void set(const uint32_t row, const uint32_t col, const value_type value);

	pointer data() noexcept;

	const_pointer data() const noexcept;

	string to_string() const;

	void print();
	void print_matrix();

	void resize(const uint32_t rows, const uint32_t cols) noexcept;

	void resize(const uint32_t size) noexcept;

	void update_dimensions(const uint32_t rows, const uint32_t cols);

	void update_dimensions(const uint32_t size);

	void reshape(const uint32_t rows, const uint32_t cols);

	void reshape(const uint32_t size);

	class ElementProxy
	{
	private:
		cumatrix<Type>& mat;
		uint32_t row;
		uint32_t col;

	public:
		/**
		 * @brief 代理类构造函数
		 * @param mat 矩阵
		 * @param row 矩阵行索引
		 * @param col 矩阵列索引
		 */
		ElementProxy(cumatrix<Type>& mat, uint32_t row, uint32_t col) : mat(mat), row(row), col(col) {}

		/**
		 * @brief 代理类析构函数
		 */
		~ElementProxy();

		/**
		 * @brief 重载类型转换运算符
		 * @return 矩阵元素值
		 */
		operator Type();
		/**
		 * @brief 重载赋值运算符
		 * @param value 待赋值的数据
		 * @return 矩阵元素值
		 */
		ElementProxy& operator=(Type value);
	};

	/**
	* @brief 重载 () 运算符，实现矩阵元素的访问和赋值
	* @param row 行索引
	* @param col 列索引
	* @return 矩阵代理类
	*/
	ElementProxy operator()(uint32_t row, uint32_t col);
private:
	uint32_t rows, cols; /// 矩阵行数和列数
	pointer mat; /// 矩阵数据
	cublasHandle_t handle; /// cuBLAS 句柄
	cusolverDnHandle_t solver_handle; /// cuSOLVER 句柄
	cudaStream_t stream; /// CUDA 流
	curandGenerator_t gen; /// cuRAND 生成器
	curandStatePhilox4_32_10* rand_states; /// cuRAND 状态
	allocator_type allocator; /// 分配器
	/**
	* @brief 分配矩阵内存
	* @param rows 矩阵行数
	* @param cols 矩阵列数
	*/
	void allocate_matrix(const uint32_t rows, const uint32_t cols);
	void deallocate_matrix() noexcept;

};
#endif // !TEMPLATE_CUDA_MATRIX_H
