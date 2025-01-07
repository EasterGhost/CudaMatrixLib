/**
* @file cuda_matrix.h
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
#define O cudaMatrix<data_type>(1)
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

#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

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

/**
* @brief CUDA 行优先矩阵类
* @class cudaMatrix<data_type>
* @note 仅支持 float 类型
* @note 使用 cuBLAS 实现部分矩阵运算
*/
template<typename data_type>
class cudaMatrix {
private:
	int rows; /// 行数
	int cols; /// 列数
	data_type* data; /// 数据指针

	/**
	* @brief 向量按指定长度扩展为矩阵
	* @param sourceVector -源向量
	* @param[in] rows 扩展后的行数
	* @param[in] cols 扩展后的列数
	* @return 结果矩阵
	* @note 支持1x1向量扩展
	*/
	static cudaMatrix<data_type> vectorBroadcast2Matrix(const cudaMatrix<data_type>& sourceVector, const int rows, const int cols);

	/**
	* @brief 矩阵点乘
	* @param[in] A 矩阵 A
	* @param[in] B 矩阵 B
	* @return 结果矩阵 A.*B
	* @throw invalid_argument 如果 A 和 B 的维度不匹配
	* @note 使用 cuBLAS 实现
	* @note 会创建新的矩阵，原矩阵不变。该方法不会释放原矩阵的内存
	*/
	static cudaMatrix<data_type> matrixDOTmatrix(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵标量乘法
	* @param[in] scalar 待乘标量
	* @return 结果矩阵 A*scalar
	* @note 使用 cuBLAS 实现
	* @note 会创建新的矩阵，原矩阵不变。该方法不会释放原矩阵的内存
	*/
	cudaMatrix<data_type> scalarMultiply(data_type scalar) const;
public:
	/**
	* @brief 默认构造函数
	* @note 仅初始化成员变量，不分配内存
	*/
	cudaMatrix();

	/**
	* @brief 构造函数
	* @param[in] rows 矩阵行数
	* @param[in] cols 矩阵列数
	*/
	cudaMatrix(int rows, int cols);

	/**
	* @brief 构造函数
	* @param[in] rows 矩阵行数
	* @param[in] cols 矩阵列数
	* @param[in] type 矩阵类型
	*/
	cudaMatrix(int rows, int cols, MatrixType type);

	/**
	* @brief 构造函数
	* @param[in] size 矩阵的阶数（行数和列数相同）
	*/
	cudaMatrix(int size);

	/**
	* @brief 构造函数
	* @param[in] size 矩阵的阶数（行数和列数相同）
	* @param[in] type 矩阵类型
	*/
	cudaMatrix(int size, MatrixType type);

	/**
	 * @brief 拷贝构造函数
	 * @param[in] other 源矩阵
	 * @note 深拷贝
	 */
	cudaMatrix(const cudaMatrix<data_type>& other);

	/**
	* @brief 析构函数
	* @note 参数置0，释放矩阵内存
	*/
	~cudaMatrix();

	static cudaMatrix<data_type> fromValue(data_type value);

	void resize(int rows, int cols);

	static cudaMatrix<data_type> zeros(int rows, int cols);

	static cudaMatrix<data_type> zeros(int size);

	static cudaMatrix<data_type> ones(int rows, int cols);

	static cudaMatrix<data_type> ones(int size);

	static cudaMatrix<data_type> identity(int size);

	static cudaMatrix<data_type> random(int rows, int cols);

	static cudaMatrix<data_type> random(int size);

	/**
	* @brief 矩阵赋值运算符重载
	* @return 结果矩阵为B
	*/
	cudaMatrix<data_type> operator = (const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵按索引赋值
	* @param[in] row 行索引
	* @param[in] col 列索引
	* @param[in] value 数据值
	*/
	void set(int row, int col, data_type value);

	/**
	* @brief 设置矩阵数据
	* @param[in] v 数据向量
	*/
	void setData(const vector<data_type>& v);

	/**
	* @brief 按索引获取矩阵数据
	* @param[in] row 行索引
	* @param[in] col 列索引
	* @return 数据值
	*/
	data_type get(int row, int col) const;

	/**
	* @brief 获取矩阵数据
	* @param[out] v 数据向量
	*/
	void getData(vector<data_type>& v, ...) const;

	/**
	* @brief 获取矩阵数据指针
	* @return 数据指针
	*/
	data_type* getDataPtr() const;

	/**
	* @brief 获取矩阵行数
	* @return 矩阵的行数
	*/
	int getRows() const;

	/**
	* @brief 获取矩阵列数
	* @return 矩阵的列数
	*/
	int getCols() const;

	/**
	* @brief 打印矩阵数据
	*/
	void printData() const;

	/**
	 * @brief 计算向量的范数
	 * @param[in] L -范数的阶数
	 * @return 向量的范数
	 */
	float norm(int L) const;

	/**
	 * @brief float 类型转换运算符重载
	 * @return float 类型数据
	 */
	operator data_type() const;

	/**
	 * @brief < 运算符重载
	 * @param n 待比较的数
	 * @param B 待比较的矩阵
	 */
	bool operator < (const data_type n);
	bool operator < (const cudaMatrix<data_type>& B);

	/**
	 * @brief <= 运算符重载
	 * @param n 待比较的数
	 */
	bool operator <= (const data_type n);

	/**
	 * @brief <= 运算符重载
	 * @param B 待比较的1x1矩阵型浮点数
	 */
	bool operator <= (const cudaMatrix<data_type>& B);

	/**
	 * @brief == 运算符重载
	 * @param n 待比较的数
	 */
	bool operator > (const data_type n);

	/**
	 * @brief == 运算符重载
	 * @param B 待比较的1x1矩阵型浮点数
	 */
	bool operator > (const cudaMatrix<data_type>& B);

	/**
	 * @brief >= 运算符重载
	 * @param n 待比较的数
	 */
	bool operator >= (const data_type n);

	/**
	 * @brief >= 运算符重载
	 * @param B 待比较的1x1矩阵型浮点数
	 */
	bool operator >= (const cudaMatrix<data_type>& B);

	void add(const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵加法静态方法
	* @param A 矩阵 A
	* @param B 矩阵 B
	* @return 结果矩阵为A+B
	* @throw invalid_argument 如果 A 和 B 的维度不匹配
	* @note 使用 cuBLAS 实现
	* @note 会创建新的矩阵，原矩阵不变。该方法不会释放原矩阵的内存
	*/
	static cudaMatrix<data_type> add(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵加法运算符重载
	* @param B 矩阵 B
	* @return 结果矩阵为A+B
	* */
	cudaMatrix<data_type> operator + (const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵加且赋值运算符重载
	* @return 结果矩阵为A+B
	*/
	cudaMatrix<data_type> operator += (const cudaMatrix<data_type>& B);

	void subtract(const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵减法静态方法
	* @param A 矩阵 A
	* @param B 矩阵 B
	* @return 结果矩阵为A-B
	*/
	static cudaMatrix<data_type> subtract(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵减法运算符重载
	* @param B 矩阵 B
	* @return 结果矩阵为A-B
	*/
	cudaMatrix<data_type> operator - (const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵减法运算符重载
	* @param B 矩阵 B
	* @return 结果矩阵为A-B
	*/
	cudaMatrix<data_type> operator -= (const cudaMatrix<data_type>& B);

	/**
	 * @brief 矩阵乘法
	 * @param B
	 * @throw invalid_argument 如果 A 的列数不等于 B 的行数
	 */
	void multiply(cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵乘法静态方法
	* @param A 矩阵 A
	* @param B 矩阵 B
	* @return 结果矩阵为A*B
	* @throw invalid_argument 如果 A 的列数不等于 B 的行数
	* @note 使用 cuBLAS 实现
	* @note 会创建新的矩阵，原矩阵不变。该方法不会释放原矩阵的内存
	*/
	static cudaMatrix<data_type> multiply(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵乘法运算符重载
	* @return 结果矩阵为A*B
	*/
	friend cudaMatrix<data_type> operator * (const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B);

	friend cudaMatrix<data_type> operator * (const data_type scalar, const cudaMatrix<data_type>& A);

	friend cudaMatrix<data_type> operator * (const cudaMatrix<data_type>& A, const data_type scalar);

	/**
	* @brief 矩阵乘且赋值运算符重载
	* @param B 矩阵 B
	* @return 结果矩阵为A*B
	*/
	cudaMatrix<data_type> operator *= (const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵标量乘且赋值运算符重载
	* @param scalar 标量
	* @return 结果矩阵为A*scalar
	*/
	cudaMatrix<data_type> operator *= (const data_type scalar);

	/**
	* @brief 矩阵幂运算符重载
	* @param pows 幂次
	* @return 结果矩阵为A^pows
	*/
	cudaMatrix<data_type> operator ^ (int pows);

	/**
	* @brief 矩阵幂且赋值运算符重载
	* @param pows 幂次
	* @return 结果矩阵为A^pows
	*/
	cudaMatrix<data_type> operator ^= (int pows);

	/**
	* @brief 矩阵转置
	* @return 转置后的矩阵
	* @note 使用 cuBLAS 实现
	* @note 会创建新的矩阵，原矩阵不变。该方法不会释放原矩阵的内存
	*/
	cudaMatrix<data_type> transpose() const;

	/**
	* @brief 矩阵转置静态方法重载
	* @param A 矩阵 A
	* @return 转置后的矩阵
	*/
	static cudaMatrix<data_type> transpose(const cudaMatrix<data_type>& A);

	/**
	* @brief 矩阵转置~运算符重载
	* @return 转置后的矩阵
	*/
	cudaMatrix<data_type> operator ~ () const;

	/**
	* @brief 矩阵求迹
	* @return 矩阵的迹
	*/
	data_type trace() const;

	/**
	* @brief 矩阵求迹静态方法重载
	* @param A 矩阵 A
	* @return 矩阵A的迹
	*/
	static data_type trace(const cudaMatrix<data_type>& A);

	/**
	* @brief 矩阵点乘
	* @param A 矩阵 A
	* @param B 矩阵 B
	* @param scalar 标量
	* @return 结果矩阵为A.*B
	* @throw invalid_argument 如果 A 和 B 的维度不匹配
	*/
	static cudaMatrix<data_type> dot(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B);
	static cudaMatrix<data_type> dot(const data_type scalar, const cudaMatrix<data_type>& A);
	static cudaMatrix<data_type> dot(const cudaMatrix<data_type>& A, const data_type scalar);

	/**
	* @brief 矩阵标量除法
	* @param A 矩阵 A
	* @param B 矩阵 B
	* @return 结果矩阵为A./B
	* @throw invalid_argument 如果 A 和 B 的维度不匹配
	*/
	static cudaMatrix<data_type> divide(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵除法运算符重载
	* @param B 矩阵 B
	* @return 结果矩阵为A./B
	* @throw invalid_argument 如果 A 和 B 的维度不匹配
	*/
	cudaMatrix<data_type> operator / (const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵标量除法运算符重载
	* @param scalar 标量
	* @return 结果矩阵为1/scalar * A
	*/
	cudaMatrix<data_type> operator / (const data_type scalar);

	/**
	* @brief 矩阵除且赋值运算符重载
	* @param B 矩阵 B
	* @return 结果矩阵为A./B
	*/
	cudaMatrix<data_type> operator /= (const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵标量除且赋值运算符重载
	* @param scalar 标量
	* @return 结果矩阵为1/scalar * A
	*/
	cudaMatrix<data_type> operator /= (const data_type scalar);

	/**
	* @brief 解稀疏线性方程组
	* @param A 系数矩阵
	* @param b 常数向量
	* @return 结果矩阵为A\b
	*/
	static cudaMatrix<data_type> solveSparseSLE(cudaMatrix<data_type>& A, cudaMatrix<data_type>& b);

	/**
	* @brief | 运算符重载为解稀疏线性方程组
	* @param b 常数向量
	* @return 结果矩阵为A\b
	*/
	cudaMatrix<data_type> operator | (cudaMatrix<data_type>& b);

	/**
	 * @brief 计算矩阵行列式
	 * @return 矩阵的行列式
	 * @throw invalid_argument 如果 A 不是方阵
	 */
	data_type det() const;

	/**
	 * @brief 计算矩阵行列式静态方法重载
	 * @param A 矩阵 A
	 * @return 矩阵A的行列式
	 * @throw invalid_argument 如果 A 不是方阵
	 */
	static data_type det(const cudaMatrix<data_type>& A);

	/**
	* @brief 生成对角矩阵
	* @param[in] placement 对角线位置
	* @param[in] ... 对角线元素向量
	* @return 对角矩阵
	* @warning placement请勿使用引用
	*/
	static cudaMatrix<data_type> diag(const vector<int> placement, ...);

	static cudaMatrix<data_type> assembleBlocks(vector<vector<cudaMatrix<data_type>>>& blocks);

	static cudaMatrix<data_type> setdiff(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B);

	/**
	* @brief 矩阵代理类
	* @note 用于重载 () 运算符，实现矩阵元素的访问和赋值
	*/
	class ElementProxy {
	private:
		cudaMatrix<data_type>& mat;
		int row;
		int col;

	public:
		/**
		 * @brief 代理类构造函数
		 * @param mat 矩阵
		 * @param row 矩阵行索引
		 * @param col 矩阵列索引
		 */
		ElementProxy(cudaMatrix<data_type>& mat, int row, int col) : mat(mat), row(row), col(col) {}

		/**
		 * @brief 代理类析构函数
		 */
		~ElementProxy();

		/**
		 * @brief 重载 float 类型转换运算符
		 * @return 矩阵元素值
		 */
		operator float() const;

		/**
		 * @brief 重载 float 类型赋值运算符
		 * @param value 待赋值的数据
		 * @return 矩阵元素值
		 */
		ElementProxy& operator=(float value);
	};

	/**
	* @brief 重载 () 运算符，实现矩阵元素的访问和赋值
	* @param row 行索引
	* @param col 列索引
	* @return 矩阵代理类
	*/
	ElementProxy operator()(int row, int col);
};
#endif // !CUDA_MATRIX_H