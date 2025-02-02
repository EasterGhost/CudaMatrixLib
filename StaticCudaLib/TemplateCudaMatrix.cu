#pragma once

#include "kernel_function.cu"
#include "kernel_function.cuh"
#include "TemplateCudaMatrix.cuh"

extern clock_t time_used_init = 0;
extern clock_t time_used_gen_init = 0;
extern clock_t time_used_gen = 0;
extern clock_t time_used_switch_type = 0;
extern clock_t time_used_setblock = 0;
extern clock_t time_used_end = 0;

template <class T>
static int autoSetBlockSize(T func)
{
	int blockSize = 0;
	int gridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, func, 0, 0);
	if (blockSize < 32)
		blockSize = 32;
	if (blockSize == 0)
		throw runtime_error("Failed to set block size.");
	return blockSize;
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
CudaMatrix<Type>::CudaMatrix(const unsigned int rows, const unsigned int cols) : rows(rows), cols(cols)
{
	int total_elements = rows * cols;
	cudaMalloc((void**)&mat, total_elements * sizeof(Type));
	cudaMemset(mat, 0, total_elements * sizeof(Type));
	cublasCreate_v2(&handle);
	cusolverDnCreate(&solver_handle);
}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const unsigned int rows, const unsigned int cols, const MatrixType type) : rows(rows), cols(cols)
{
	clock_t start = clock();
	size_t total_elements = static_cast<size_t>(rows) * cols;
	cudaMalloc((void**)&mat, total_elements * sizeof(Type));
	int blockSize = 0;
	int gridSize = 0;
	curandStatePhilox4_32_10_t* states = nullptr;
	curandStateScrambledSobol32_t* qstates32 = nullptr;
	curandStateScrambledSobol64_t* qstates64 = nullptr;
	curandDirectionVectors32_t* dr_vec32 = nullptr;
	curandDirectionVectors64_t* dr_vec64 = nullptr;
	cudaDeviceSynchronize();
	time_used_init += clock() - start;
	switch (type)
	{
	case Zero:
		cudaMemset(mat, 0, total_elements * sizeof(Type));
		break;
	case Ones:
		blockSize = autoSetBlockSize(ones_matrix_kernel<Type>);
		gridSize = (total_elements + blockSize - 1) / blockSize;
		ones_matrix_kernel<Type> << <gridSize, blockSize >> > (mat, total_elements);
		cudaDeviceSynchronize();
		break;
	case Identity:
		if (rows != cols)
			throw runtime_error("Identity matrix must be square matrix.");
		cudaMemset(mat, 0, static_cast<size_t>(rows) * cols * sizeof(Type));
		blockSize = autoSetBlockSize(identity_matrix_kernel<Type>);
		gridSize = (rows + blockSize - 1) / blockSize;
		identity_matrix_kernel<Type> << <gridSize, blockSize >> > (mat, rows);
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
		if constexpr (is_same<Type, float>::value)
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
		else if constexpr (is_same<Type, double>::value)
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
		else if constexpr (is_same<Type, int>::value)
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
			total_elements = static_cast<size_t>(rows) * cols * sizeof(Type) / sizeof(int);
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
	case QuasiRandom:
		start = clock();
		if constexpr (is_same<Type, float>::value)
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
			float_qrandom_matrix_kernel << <gridSize, blockSize >> >((float*)mat, qstates32, rows, cols);
			cudaDeviceSynchronize();
			time_used_gen += clock() - start;
			start = clock();
			cudaFree(qstates32);
			cudaFree(dr_vec32);
		}
		else if constexpr (is_same<Type, double>::value)
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
		else if constexpr (is_same<Type, int>::value)
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
			int_qrandom_matrix_kernel << <gridSize, blockSize >> >((int*)mat, qstates32, rows, cols);
			cudaDeviceSynchronize();
			time_used_gen += clock() - start;
			start = clock();
			cudaFree(qstates32);
			cudaFree(dr_vec32);
		}
		else
		{
			total_elements = static_cast<size_t>(rows) * cols * sizeof(Type) / sizeof(int);
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
	default:
		throw runtime_error("Unknown matrix type.");
	}
	cudaDeviceSynchronize();
	cublasCreate_v2(&handle);
	cusolverDnCreate(&solver_handle);
	time_used_end += clock() - start;
}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const unsigned int size) : CudaMatrix(size, size) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const unsigned int size, const MatrixType type) : CudaMatrix(size, size, type) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const unsigned int rows, const unsigned int cols, const Type* src) : CudaMatrix(rows, cols) { cudaMemcpy(mat, src, static_cast<size_t>(rows) * cols * sizeof(Type), cudaMemcpyHostToDevice); }

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const unsigned int size, const Type* src) : CudaMatrix(size, size, src) {}

template<typename Type>
CudaMatrix<Type>::CudaMatrix(const unsigned int size, const vector<Type>& src) : CudaMatrix(size, size, src.data()) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const unsigned int rows, unsigned int cols, const vector<Type>& src) : CudaMatrix(rows, cols, src.data()) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(const CudaMatrix<Type>& other) : rows(other.rows), cols(other.cols)
{
	int total_elements = rows * cols;
	cudaMalloc((void**)&mat, total_elements * sizeof(Type));
	cudaMemcpy(mat, other.mat, total_elements * sizeof(Type), cudaMemcpyDeviceToDevice);
	cublasCreate_v2(&handle);
	cusolverDnCreate(&solver_handle);
}

template <typename Type>
CudaMatrix<Type>::~CudaMatrix()
{
	cudaMemset(mat, 0, static_cast<size_t>(rows) * cols * sizeof(Type));
	cudaFree(mat);
	cublasDestroy_v2(handle);
	cusolverDnDestroy(solver_handle);
	rows = 0;
	cols = 0;
	mat = nullptr;
}

template<typename Type>
void CudaMatrix<Type>::set(const unsigned int row, const unsigned int col, const Type value)
{
	if (row >= rows || col >= cols)
		throw out_of_range("Index out of range.");
	CUDA_CHECK(cudaMemcpy(mat + row * cols + col, &value, sizeof(Type), cudaMemcpyHostToDevice));
}

template<typename Type>
Type* CudaMatrix<Type>::data() const { return this->mat; }

template<typename Type>
void CudaMatrix<Type>::print()
{
	Type* host_data = new Type[rows * cols];
	cudaMemcpy(host_data, mat, static_cast<size_t>(rows) * cols * sizeof(Type), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			cout << host_data[i * cols + j] << " ";
		cout << endl;
	}
	delete[] host_data;
}

template<typename Type>
unsigned int CudaMatrix<Type>::getRows() const { return rows; }

template<typename Type>
unsigned int CudaMatrix<Type>::getCols() const { return cols; }

template<typename Type>
void CudaMatrix<Type>::getData(Type* dst) const { cudaMemcpy(dst, mat, static_cast<size_t>(rows) * cols * sizeof(Type), cudaMemcpyDeviceToHost); }

template<typename Type>
void CudaMatrix<Type>::setData(const vector<Type>& src) { cudaMemcpy(mat, src.data(), src.size(), cudaMemcpyHostToDevice); }

template<typename Type>
Type CudaMatrix<Type>::get(const unsigned int row, const unsigned int col) const
{
	if (row >= rows || col >= cols)
		throw out_of_range("Index out of range.");
	Type res = 0;
	cudaMemcpy(&res, mat + row * cols + col, sizeof(Type), cudaMemcpyDeviceToHost);
	return res;
}

template<typename Type>
template<typename T>
void CudaMatrix<Type>::add(const CudaMatrix<T>& other)
{
	if (rows != other.rows || cols != other.cols)
		throw runtime_error("Matrix size does not match.");
	int total_elements = rows * cols;
	int blockSize = autoSetBlockSize(elementwise_add_kernel<Type, T, Type>);
	int gridSize = (total_elements + blockSize - 1) / blockSize;
	elementwise_add_kernel<Type, T, Type> << <gridSize, blockSize >> >
		(mat, other.mat, mat, total_elements);
}