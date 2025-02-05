#pragma once

#include "kernel_function.cu"
#include "kernel_function.cuh"
#include "cuda_matrix.cuh"

template <class T>
static int autoSetBlockSize(T func)
{
	int blockSize = 0;
	int gridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, func, 0, 0);
	//if (blockSize == 0)
	//	throw runtime_error("Failed to set block size.");
	return max(blockSize, 32);
}

template <class T>
static dim3 autoSetBlockSize2D(T func, const int rows, const int cols)
{
	int blockSize = 0;
	int gridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, func, 0, 0);
	if (blockSize == 0)
		throw runtime_error("Failed to set block size.");
	return dim3(blockSize, 1);
}

template <typename Type>
CudaMatrix<Type>::CudaMatrix()
{
	rows = 0;
	cols = 0;
	mat = nullptr;
}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const uint32_t rows, const uint32_t cols) : rows(rows), cols(cols)
{
	int total_elements = rows * cols;
	cudaMalloc((void**)&mat, total_elements * sizeof(value_type));
	cudaMemset(mat, 0, total_elements * sizeof(value_type));
	cublasCreate_v2(&handle);
	cusolverDnCreate(&solver_handle);
}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const uint32_t rows, const uint32_t cols, const MatrixType type) : rows(rows), cols(cols)
{
	clock_t start = clock();
	size_t total_elements = static_cast<size_t>(rows) * cols;
	cudaMalloc((void**)&mat, total_elements * sizeof(value_type));
	int blockSize = 0;
	int gridSize = 0;
	curandStatePhilox4_32_10_t* states = nullptr;
	cudaDeviceSynchronize();
	time_used_init += clock() - start;
	switch (type)
	{
	case Zero:
		cudaMemset(mat, 0, total_elements * sizeof(value_type));
		break;
	case Ones:
		blockSize = autoSetBlockSize(ones_matrix_kernel<value_type>);
		gridSize = (total_elements + blockSize - 1) / blockSize;
		ones_matrix_kernel<value_type> << <gridSize, blockSize >> > (mat, total_elements);
		cudaDeviceSynchronize();
		break;
	case Identity:
		if (rows != cols)
			throw runtime_error("Identity matrix must be square matrix.");
		cudaMemset(mat, 0, static_cast<size_t>(rows) * cols * sizeof(value_type));
		blockSize = autoSetBlockSize(identity_matrix_kernel<value_type>);
		gridSize = (rows + blockSize - 1) / blockSize;
		identity_matrix_kernel<value_type> << <gridSize, blockSize >> > (mat, rows);
		cudaDeviceSynchronize();
		break;
	case Random:
		start = clock();
		blockSize = autoSetBlockSize(setup_random_kernel);
		//cout << "Block size of setup random kernel: " << blockSize << endl;
		gridSize = (total_elements + blockSize - 1) / blockSize;
		cudaMalloc((void**)&states, total_elements * sizeof(curandStatePhilox4_32_10_t));
		setup_random_kernel << <gridSize, blockSize >> > (states, time(0), total_elements);
		cudaDeviceSynchronize();
		time_used_gen_init += clock() - start;
		start = clock();
		if constexpr (is_same<value_type, float>::value)
		{
			time_used_switch_type += clock() - start;
			start = clock();
			blockSize = autoSetBlockSize(float_random_matrix_kernel);
			//cout << "Block size of float random kernel: " << blockSize << endl;
			gridSize = (total_elements + blockSize - 1) / blockSize;
			time_used_setblock += clock() - start;
			start = clock();
			float_random_matrix_kernel << <gridSize, blockSize >> >
				((float*)mat, total_elements, states);
			cudaDeviceSynchronize();
			time_used_gen += clock() - start;
			start = clock();
		}
		else if constexpr (is_same<value_type, double>::value)
		{
			time_used_switch_type += clock() - start;
			start = clock();
			blockSize = autoSetBlockSize(double_random_matrix_kernel);
			//cout << "Block size of double random kernel: " << blockSize << endl;
			gridSize = (total_elements + blockSize - 1) / blockSize;
			time_used_setblock += clock() - start;
			start = clock();
			double_random_matrix_kernel << <gridSize, blockSize >> >
				((double*)mat, total_elements, states);
			cudaDeviceSynchronize();
			time_used_gen += clock() - start;
			start = clock();
		}
		else if constexpr (is_same<value_type, int>::value)
		{
			time_used_switch_type += clock() - start;
			start = clock();
			blockSize = autoSetBlockSize(int_random_matrix_kernel);
			//cout << "Block size of int random kernel: " << blockSize << endl;
			gridSize = (total_elements + blockSize - 1) / blockSize;
			time_used_setblock += clock() - start;
			start = clock();
			int_random_matrix_kernel << <gridSize, blockSize >> >
				((int*)mat, total_elements, states);
			cudaDeviceSynchronize();
			time_used_gen += clock() - start;
			start = clock();
		}
		else
		{
			total_elements = static_cast<size_t>(rows) * cols * sizeof(value_type) / sizeof(int);
			//cudaFree(states);
			cudaMalloc((void**)&states, total_elements * sizeof(curandStatePhilox4_32_10_t));
			setup_random_kernel << <gridSize, blockSize >> > (states, time(0), total_elements);
			cudaDeviceSynchronize();
			time_used_gen_init += clock() - start;
			start = clock();
			blockSize = autoSetBlockSize(int_random_matrix_kernel);
			//cout << "Block size of int random kernel: " << blockSize << endl;
			gridSize = (total_elements + blockSize - 1) / blockSize;
			time_used_setblock += clock() - start;
			start = clock();
			int_random_matrix_kernel << <gridSize, blockSize >> >
				((int*)mat, total_elements, states);
			cudaDeviceSynchronize();
			time_used_gen += clock() - start;
			start = clock();
		}
		cudaFree(states);
		break;
		/*
	case QuasiRandom:
		start = clock();
		if constexpr (is_same<value_type, float>::value)
		{
			blockSize = autoSetBlockSize(setup_q32random_kernel);
			//cout << "Block size of setup random kernel: " << blockSize << endl;
			gridSize = (total_elements + blockSize - 1) / blockSize;
			cudaMalloc((void**)&qstates32, total_elements * sizeof(curandStateScrambledSobol32_t));
			cudaMalloc((void**)&dr_vec32, cols * sizeof(curandDirectionVectors32_t));
			curandGetDirectionVectors32(&dr_vec32, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
			setup_q32random_kernel << <gridSize, blockSize >> > (qstates32, dr_vec32, rows, cols);
			cudaDeviceSynchronize();
			time_used_gen_init += clock() - start;
			start = clock();
			blockSize = autoSetBlockSize(float_qrandom_matrix_kernel);
			//cout << "Block size of float random kernel: " << blockSize << endl;
			gridSize = (total_elements + blockSize - 1) / blockSize;
			time_used_setblock += clock() - start;
			start = clock();
			float_qrandom_matrix_kernel << <gridSize, blockSize >> > ((float*)mat, qstates32, rows, cols);
			cudaDeviceSynchronize();
			time_used_gen += clock() - start;
			start = clock();
			cudaFree(qstates32);
			cudaFree(dr_vec32);
		}
		else if constexpr (is_same<value_type, double>::value)
		{
			blockSize = autoSetBlockSize(setup_q64random_kernel);
			//cout << "Block size of setup random kernel: " << blockSize << endl;
			gridSize = (total_elements + blockSize - 1) / blockSize;
			cudaMalloc((void**)&qstates64, total_elements * sizeof(curandStateScrambledSobol64_t));
			cudaMalloc((void**)&dr_vec64, cols * sizeof(curandDirectionVectors64_t));
			curandGetDirectionVectors64(&dr_vec64, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6);
			setup_q64random_kernel << <gridSize, blockSize >> > (qstates64, dr_vec64, rows, cols);
			cudaDeviceSynchronize();
			time_used_gen_init += clock() - start;
			start = clock();
			blockSize = autoSetBlockSize(double_qrandom_matrix_kernel);
			//cout << "Block size of double random kernel: " << blockSize << endl;
			gridSize = (total_elements + blockSize - 1) / blockSize;
			time_used_setblock += clock() - start;
			start = clock();
			double_qrandom_matrix_kernel << <gridSize, blockSize >> >
				((double*)mat, qstates64, rows, cols);
			cudaDeviceSynchronize();
			time_used_gen += clock() - start;
			start = clock();
			cudaFree(qstates64);
			cudaFree(dr_vec64);
		}
		else if constexpr (is_same<value_type, int>::value)
		{
			blockSize = autoSetBlockSize(setup_q32random_kernel);
			//cout << "Block size of setup random kernel: " << blockSize << endl;
			//system("pause");
			gridSize = (total_elements + blockSize - 1) / blockSize;
			cudaMalloc((void**)&qstates32, total_elements * sizeof(curandStateScrambledSobol32_t));
			cudaMalloc((void**)&dr_vec32, cols * sizeof(curandDirectionVectors32_t));
			curandGetDirectionVectors32(&dr_vec32, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
			setup_q32random_kernel << <gridSize, blockSize >> > (qstates32, dr_vec32, rows, cols);
			cudaDeviceSynchronize();
			time_used_gen_init += clock() - start;
			start = clock();
			blockSize = autoSetBlockSize(int_qrandom_matrix_kernel);
			cout << "Block size of int random kernel: " << blockSize << endl;
			gridSize = (total_elements + blockSize - 1) / blockSize;
			time_used_setblock += clock() - start;
			start = clock();
			int_qrandom_matrix_kernel << <gridSize, blockSize >> > ((int*)mat, qstates32, rows, cols);
			cudaDeviceSynchronize();
			time_used_gen += clock() - start;
			start = clock();
			cudaFree(qstates32);
			cudaFree(dr_vec32);
		}
		else
		{
			total_elements = static_cast<size_t>(rows) * cols * sizeof(value_type) / sizeof(int);
			cudaMalloc((void**)&qstates32, total_elements * sizeof(curandStateScrambledSobol32_t));
			cudaMalloc((void**)&dr_vec32, cols * sizeof(curandDirectionVectors32_t));
			curandGetDirectionVectors32(&dr_vec32, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
			setup_q32random_kernel << <gridSize, blockSize >> > (qstates32, dr_vec32, rows, cols);
			cudaDeviceSynchronize();
			time_used_gen_init += clock() - start;
			start = clock();
			blockSize = autoSetBlockSize(int_qrandom_matrix_kernel);
			//cout << "Block size of int random kernel: " << blockSize << endl;
			gridSize = (total_elements + blockSize - 1) / blockSize;
			time_used_setblock += clock() - start;
			start = clock();
			int_qrandom_matrix_kernel << <gridSize, blockSize >> >
				((int*)mat, qstates32, rows, cols);
			cudaDeviceSynchronize();
			time_used_gen += clock() - start;
			start = clock();
			cudaFree(qstates32);
			cudaFree(dr_vec32);
		}
		break;
		*/
	default:
		throw runtime_error("Unknown matrix type.");
	}
	cudaDeviceSynchronize();
	cublasCreate_v2(&handle);
	cusolverDnCreate(&solver_handle);
	time_used_end += clock() - start;
}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const uint32_t size) : CudaMatrix(size, size) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const uint32_t size, const MatrixType type) : CudaMatrix(size, size, type) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const uint32_t rows, const uint32_t cols, const pointer src) : CudaMatrix(rows, cols) { cudaMemcpy(mat, src, static_cast<size_t>(rows) * cols * sizeof(value_type), cudaMemcpyHostToDevice); }

template<typename Type>
CudaMatrix<Type>::CudaMatrix(const uint32_t rows, const uint32_t cols, const vector<value_type>& src) : CudaMatrix(rows, cols, src.data()) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const uint32_t size, const pointer src) : CudaMatrix(size, size, src) {}

template<typename Type>
CudaMatrix<Type>::CudaMatrix(const uint32_t size, const vector<value_type>& src) : CudaMatrix(size, size, src.data()) {}

template<typename Type>
CudaMatrix<Type>::CudaMatrix(CudaMatrix&& other) noexcept
{
	rows = other.rows;
	cols = other.cols;
	mat = other.mat;
	other.rows = 0;
	other.cols = 0;
	other.mat = nullptr;
	handle = other.handle;
	solver_handle = other.solver_handle;
	other.handle = nullptr;
	other.solver_handle = nullptr;
}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const CudaMatrix<value_type>& other) : rows(other.rows), cols(other.cols)
{
	int total_elements = rows * cols;
	cudaMalloc((void**)&mat, total_elements * sizeof(value_type));
	cudaMemcpy(mat, other.mat, total_elements * sizeof(value_type), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	cublasCreate_v2(&handle);
	cusolverDnCreate(&solver_handle);
}

template<typename Type>
CudaMatrix<Type>& CudaMatrix<Type>::operator=(const CudaMatrix<value_type>& other)
{
	if (this == &other)
		return *this;
	if (mat != nullptr)
		cudaFree(mat);
	rows = other.rows;
	cols = other.cols;
	cudaMalloc((void**)&mat, static_cast<size_t>(rows) * cols * sizeof(value_type));
	cudaMemcpy(mat, other.mat, static_cast<size_t>(rows) * cols * sizeof(value_type), cudaMemcpyDeviceToDevice);
	cublasCreate_v2(&handle);
	cusolverDnCreate(&solver_handle);
	return *this;
}

template<typename Type>
CudaMatrix<Type>& CudaMatrix<Type>::operator=(CudaMatrix&& other) noexcept
{
	if (this == &other)
		return *this;
	if (mat != nullptr)
		cudaFree(mat);
	rows = other.rows;
	cols = other.cols;
	mat = other.mat;
	other.rows = 0;
	other.cols = 0;
	other.mat = nullptr;
	handle = other.handle;
	solver_handle = other.solver_handle;
	other.handle = nullptr;
	other.solver_handle = nullptr;
	return *this;
}

template <typename Type>
CudaMatrix<Type>::~CudaMatrix()
{
	if (IS_SAFE_DATA) cudaMemset(mat, 0, static_cast<size_t>(rows) * cols * sizeof(value_type));
	cudaFree(mat);
	rows = 0;
	cols = 0;
	mat = nullptr;
	cudaDeviceSynchronize();
	//cublasDestroy_v2(handle);
	//cusolverDnDestroy(solver_handle);
}

template<typename Type>
bool CudaMatrix<Type>::operator==(const CudaMatrix<value_type>& other) const
{
	if (rows != other.rows || cols != other.cols)
		return false;
	int total_elements = rows * cols;
	int blockSize = autoSetBlockSize(elementwise_equal_kernel<value_type>);
	int gridSize = (total_elements + blockSize - 1) / blockSize;
	char* res = nullptr;
	cudaMalloc((void**)&res, total_elements * sizeof(char));
	elementwise_equal_kernel<value_type> << <gridSize, blockSize >> >
		(mat, other.mat, res, total_elements);
	cudaDeviceSynchronize();
	vector<char> host_res(total_elements);
	cudaMemcpy(host_res.data(), res, total_elements * sizeof(char), cudaMemcpyDeviceToHost);
	bool result = all_of(host_res.begin(), host_res.end(), [](char x) { return x; });
	cudaFree(res);
	return result;
}

template<typename Type>
bool CudaMatrix<Type>::operator!=(const CudaMatrix<value_type>& other) const
{
	if (rows != other.rows || cols != other.cols)
		return true;
	int total_elements = rows * cols;
	int blockSize = autoSetBlockSize(elementwise_equal_kernel<value_type>);
	int gridSize = (total_elements + blockSize - 1) / blockSize;
	char* res = nullptr;
	cudaMalloc((void**)&res, total_elements * sizeof(char));
	elementwise_equal_kernel<value_type> << <gridSize, blockSize >> >
		(mat, other.mat, res, total_elements);
	cudaDeviceSynchronize();
	vector<char> host_res(total_elements);
	cudaMemcpy(host_res.data(), res, total_elements * sizeof(bool), cudaMemcpyDeviceToHost);
	bool result = any_of(host_res.begin(), host_res.end(), [](char x) { return !x; });
	cudaFree(res);
	return result;
}

template<typename Type>
size_t CudaMatrix<Type>::size() const noexcept { return static_cast<size_t>(rows) * cols; }

template<typename Type>
bool CudaMatrix<Type>::empty() const noexcept { return (mat == nullptr || size() == 0); }

template<typename Type>
constexpr size_t CudaMatrix<Type>::max_size() const noexcept { return numeric_limits<size_t>::max() / sizeof(value_type); }

template<typename Type>Type& CudaMatrix<Type>::front()
{
	if (empty())
		throw runtime_error("Matrix is empty.");
	value_type res = 0;
	cudaMemcpy(&res, mat, sizeof(value_type), cudaMemcpyDeviceToHost);
	return res;
}

template<typename Type>
const Type& CudaMatrix<Type>::front() const
{
	if (empty())
		throw runtime_error("Matrix is empty.");
	value_type res = 0;
	cudaMemcpy(&res, mat, sizeof(value_type), cudaMemcpyDeviceToHost);
	return res;
}

template<typename Type>
Type& CudaMatrix<Type>::back()
{
	if (empty())
		throw runtime_error("Matrix is empty.");
	value_type res = 0;
	cudaMemcpy(&res, mat + static_cast<size_t>(rows) * cols - 1, sizeof(value_type), cudaMemcpyDeviceToHost);
	return res;
}

template<typename Type>
const Type& CudaMatrix<Type>::back() const
{
	if (empty())
		throw runtime_error("Matrix is empty.");
	value_type res = 0;
	cudaMemcpy(&res, mat + static_cast<size_t>(rows) * cols - 1, sizeof(value_type), cudaMemcpyDeviceToHost);
	return res;
}

template<typename Type>
Type CudaMatrix<Type>::operator[](const coord_t coord) const
{
	if (coord.x >= rows || coord.y >= cols)
		throw out_of_range("Index out of range.");
	value_type res = 0;
	cudaMemcpy(&res, mat + coord.x * rows + coord.y, sizeof(value_type), cudaMemcpyDeviceToHost);
	return res;
}

template<typename Type>
Type CudaMatrix<Type>::operator[](const uint32_t index) const
{
	if (index >= static_cast<size_t>(rows) * cols)
		throw out_of_range("Index out of range.");
	value_type res = 0;
	cudaMemcpy(&res, mat + index, sizeof(value_type), cudaMemcpyDeviceToHost);
	return res;
}

template<typename Type>
void CudaMatrix<Type>::clear() noexcept
{
	if (IS_SAFE_DATA) cudaMemset(mat, 0, static_cast<size_t>(rows) * cols * sizeof(value_type));
	cudaFree(mat);
	rows = 0;
	cols = 0;
	mat = nullptr;
}

template<typename Type>
void CudaMatrix<Type>::swap(CudaMatrix<value_type>& other) noexcept
{
	swap(rows, other.rows);
	swap(cols, other.cols);
	swap(mat, other.mat);
}

template<typename Type>
void CudaMatrix<Type>::assign(const CudaMatrix<value_type>& other)
{
	if (this == &other)
		return;
	rows = other.rows;
	cols = other.cols;
	if (mat != nullptr)
		cudaFree(mat);
	cudaMalloc((void**)&mat, static_cast<size_t>(rows) * cols * sizeof(value_type));
	cudaMemcpy(mat, other.mat, static_cast<size_t>(rows) * cols * sizeof(value_type), cudaMemcpyDeviceToDevice);
}

template<typename Type>
void CudaMatrix<Type>::assign(CudaMatrix<value_type>&& other) noexcept
{
	if (this == &other)
		return;
	if (mat != nullptr)
		cudaFree(mat);
	rows = other.rows;
	cols = other.cols;
	mat = other.mat;
	other.rows = 0;
	other.cols = 0;
	other.mat = nullptr;
}

template<typename Type>
void CudaMatrix<Type>::assign(const uint32_t rows, const uint32_t cols, const_reference val)
{
	if (mat != nullptr)
		cudaFree(mat);
	this->rows = rows;
	this->cols = cols;
	cudaMalloc((void**)&mat, static_cast<size_t>(rows) * cols * sizeof(value_type));
	int blockSize = autoSetBlockSize(assign_kernel<value_type>);
	int gridSize = (rows * cols + blockSize - 1) / blockSize;
	assign_kernel<value_type> << <gridSize, blockSize >> >
		(mat, val, rows * cols);
	cudaDeviceSynchronize();
}

template<typename Type>
void CudaMatrix<Type>::assign(const uint32_t size, const_reference val) { assign(size, size, val); }

template<typename Type>
void CudaMatrix<Type>::assign(const initializer_list<value_type>& il)
{
	if (mat != nullptr)
		cudaFree(mat);
	rows = il.size();
	cols = 1;
	cudaMalloc((void**)&mat, static_cast<size_t>(rows) * cols * sizeof(value_type));
	cudaMemcpy(mat, il.begin(), static_cast<size_t>(rows) * cols * sizeof(value_type), cudaMemcpyHostToDevice);
}

template<typename Type>
void CudaMatrix<Type>::insert(const uint32_t rows, const uint32_t cols, const_reference val)
{
	if (rows == 0 || cols == 0)
		throw runtime_error("Invalid matrix size.");
	if (this->rows * this->cols < rows * cols)
	{
		resize(rows, cols);
		set(rows, cols, val);
		return;
	}
}

template<typename Type>
Type CudaMatrix<Type>::at(const uint32_t rows, const uint32_t cols) const
{
	if (rows >= this->rows || cols >= this->cols)
		throw out_of_range("Index out of range.");
	value_type res = 0;
	cudaMemcpy(&res, mat + rows * this->cols + cols, sizeof(value_type), cudaMemcpyDeviceToHost);
	return res;
}

template<typename Type>
size_t CudaMatrix<Type>::capacity() const noexcept { return static_cast<size_t>(rows) * cols; }

template<typename Type>
void CudaMatrix<Type>::set(const uint32_t row, const uint32_t col, const value_type value)
{
	if (row >= rows || col >= cols)
		throw out_of_range("Index out of range.");
	cudaMemcpy(mat + row * cols + col, &value, sizeof(value_type), cudaMemcpyHostToDevice);
}

template<typename Type>
Type* CudaMatrix<Type>::data() noexcept { return this->mat; }

template<typename Type>
const Type* CudaMatrix<Type>::data() const noexcept { return const_cast<pointer>(this->mat); }

template<typename Type>
void CudaMatrix<Type>::print_matrix()
{
	pointer host_data = new value_type[rows * cols];
	cudaMemcpy(host_data, mat, static_cast<size_t>(rows) * cols * sizeof(value_type), cudaMemcpyDeviceToHost);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			cout << host_data[i * cols + j] << " ";
		cout << endl;
	}
	if (IS_SAFE_DATA) memset(host_data, 0, static_cast<size_t>(rows) * cols * sizeof(value_type));
	delete[] host_data;
}

template<typename Type>
string CudaMatrix<Type>::to_string() const
{
	pointer host_data = new value_type[rows * cols];
	cudaMemcpy(host_data, mat, static_cast<size_t>(rows) * cols * sizeof(value_type), cudaMemcpyDeviceToHost);
	string res;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			res += to_string(host_data[i * cols + j]) + " ";
		res += "\n";
	}
	if (IS_SAFE_DATA) memset(host_data, 0, static_cast<size_t>(rows) * cols * sizeof(value_type));
	delete[] host_data;
	return res;
}

template<typename Type>
void CudaMatrix<Type>::print() { print_matrix(); }

template<typename Type>
void CudaMatrix<Type>::resize(const uint32_t rows, const uint32_t cols) noexcept
{
	if (this->rows == rows && this->cols == cols)
		return;
	try
	{
		if (mat != nullptr)
			cudaFree(mat);
		this->rows = rows;
		this->cols = cols;
		cudaError_t err = cudaMalloc((void**)&mat, static_cast<size_t>(rows) * cols * sizeof(value_type));
		err += cudaMemset(mat, 0, static_cast<size_t>(rows) * cols * sizeof(value_type));
		if (err != cudaSuccess)
			throw runtime_error("Failed to resize matrix.");
	}
	catch (const exception& err) {
		cerr << err.what() << endl;
	}
}

template<typename Type>
void CudaMatrix<Type>::resize(const uint32_t size) noexcept { resize(size, size); }

template<typename Type>
void CudaMatrix<Type>::update_dimensions(const uint32_t rows, const uint32_t cols)
{
	if (this->rows * this->cols != rows * cols)
		throw runtime_error("The number of elements in the matrix does not match the new shape.");
	this->rows = rows;
	this->cols = cols;
}

template<typename Type>
void CudaMatrix<Type>::update_dimensions(const uint32_t size) { update_dimensions(size, size); }

template<typename Type>
void CudaMatrix<Type>::reshape(const uint32_t rows, const uint32_t cols)
{
	if (this->rows == rows && this->cols == cols)
		return;
	pointer tmp = nullptr;
	cudaMalloc((void**)&tmp, static_cast<size_t>(rows) * cols * sizeof(value_type));
	cudaMemset(tmp, 0, static_cast<size_t>(rows) * cols * sizeof(value_type));
	dim3 blockSize = autoSetBlockSize2D(reshape_kernel<value_type>);
	dim3 gridSize = dim3((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);
	reshape_kernel<value_type> << <gridSize, blockSize >> >
		(mat, tmp, rows, cols, row, col);
	cudaDeviceSynchronize();
	cudaFree(mat);
	mat = tmp;
	this->rows = row;
	this->cols = col;
	tmp = nullptr;
}

template<typename Type>
void CudaMatrix<Type>::reshape(const uint32_t size) { reshape(size, size); }

template<typename Type>
CudaMatrix<Type>::ElementProxy CudaMatrix<Type>::operator()(uint32_t row, uint32_t col)
{
	if (row >= rows || col >= cols)
		throw out_of_range("Index out of range.");
	return ElementProxy(*this, row, col);
}

template<typename Type>
uint32_t CudaMatrix<Type>::row_count() const { return rows; }

template<typename Type>
uint32_t CudaMatrix<Type>::col_count() const { return cols; }

template<typename Type>
void CudaMatrix<Type>::get_data(pointer dst) const { cudaMemcpy(dst, mat, _msize(dst) * sizeof(value_type), cudaMemcpyDeviceToHost); }

template<typename Type>
void CudaMatrix<Type>::get_data(vector<value_type>& dst) const { cudaMemcpy(dst.data(), mat, dst.size() * sizeof(value_type), cudaMemcpyDeviceToHost); }

template<typename Type>
void CudaMatrix<Type>::get_data(vector<value_type>& dst, bool is_safesize) const
{
	if (is_safesize && dst.size() < static_cast<size_t>(rows) * cols)
		dst.resize(static_cast<size_t>(rows) * cols);
	cudaMemcpy(dst.data(), mat, static_cast<size_t>(rows) * cols * sizeof(Type), cudaMemcpyDeviceToHost);
}

template<typename Type>
void CudaMatrix<Type>::set_data(const pointer src) { cudaMemcpy(mat, src, _msize(src) * sizeof(value_type), cudaMemcpyHostToDevice); }

template<typename Type>
void CudaMatrix<Type>::set_data(const vector<value_type>& src) { cudaMemcpy(mat, src.data(), src.size(), cudaMemcpyHostToDevice); }

template<typename Type>
Type CudaMatrix<Type>::get(const uint32_t row, const uint32_t col) const
{
	if (row >= rows || col >= cols)
		throw out_of_range("Index out of range.");
	value_type res = 0;
	cudaMemcpy(&res, mat + row * cols + col, sizeof(value_type), cudaMemcpyDeviceToHost);
	return res;
}

template<typename Type>
CudaMatrix<Type>::ElementProxy::~ElementProxy()
{
	row = 0;
	col = 0;
}

template<typename Type>
CudaMatrix<Type>::ElementProxy::operator Type() { return mat.get(row, col); }

template<typename Type>
CudaMatrix<Type>::ElementProxy& CudaMatrix<Type>::ElementProxy::operator=(Type value)
{
	mat.set(row, col, value);
	return *this;
}
