#include "TemplateCudaMatrix.cuh"
#include "kernel_function.cuh"
#include "kernel_function.cu"

template <class T>
static int autoSetBlockSize(T func)
{
	int blockSize = 0;
	int gridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, func, 0, 0);
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
CudaMatrix<Type>::CudaMatrix(int rows, int cols) : rows(rows), cols(cols)
{
	int total_elements = rows * cols;
	cudaMalloc((void**)&mat, total_elements * sizeof(Type));
	cudaMemset(mat, 0, total_elements * sizeof(Type));
	cublasCreate_v2(&handle);
	cusolverDnCreate(&solver_handle);
}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(int rows, int cols, MatrixType type) : rows(rows), cols(cols)
{
	int total_elements = rows * cols;
	cudaMalloc((void**)&mat, total_elements * sizeof(Type));
	int blockSize = 0;
	int gridSize = 0;
	curandState* states = nullptr;

	switch (type)
	{
	case Zero:
		cudaMemset(mat, 0, total_elements * sizeof(Type));
		break;
	case Ones:
		blockSize = autoSetBlockSize(ones_matrix_kernel<Type>);
		gridSize = (total_elements + blockSize - 1) / blockSize;
		ones_matrix_kernel<Type> << <gridSize, blockSize >> > (mat, total_elements);
		break;
	case Identity:
		if (rows != cols)
		{
			throw runtime_error("Identity matrix must be square matrix.");
		}
		cudaMemset(mat, 0, static_cast<size_t>(rows) * cols * sizeof(Type));
		blockSize = autoSetBlockSize(identity_matrix_kernel<Type>);
		gridSize = (rows + blockSize - 1) / blockSize;
		identity_matrix_kernel<Type> << <gridSize, blockSize >> > (mat, rows);
		break;
	case Random:
		blockSize = autoSetBlockSize(random_matrix_kernel<Type>);
		gridSize = (total_elements + blockSize - 1) / blockSize;
		cudaMalloc((void**)&states, total_elements * sizeof(curandState));
		setup_random_kernel << <gridSize, blockSize >> > (states, time(0), total_elements);
		random_matrix_kernel<Type> << <gridSize, blockSize >> > (mat, total_elements, states);

		break;
	default:
		throw runtime_error("Unknown matrix type.");
	}
	cublasCreate_v2(&handle);
	cusolverDnCreate(&solver_handle);
	cudaFree(states);
}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(int size) : CudaMatrix(size, size) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(int size, MatrixType type) : CudaMatrix(size, size, type) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(int rows, int cols, Type* src) : CudaMatrix(rows, cols) { cudaMemcpy(mat, src, static_cast<size_t>(rows) * cols * sizeof(Type), cudaMemcpyHostToDevice); }

template <typename Type>
CudaMatrix<Type>::CudaMatrix(int size, Type* src) : CudaMatrix(size, size, src) {}

template<typename Type>
CudaMatrix<Type>::CudaMatrix(int size, vector<Type> src) : CudaMatrix(size, size, src.data()) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(int rows, int cols, vector<Type> src) : CudaMatrix(rows, cols, src.data()) {}

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
	rows = 0;
	cols = 0;
	cudaFree(mat);
	mat = nullptr;
	cublasDestroy_v2(handle);
	cusolverDnDestroy(solver_handle);
}

template<typename Type>
void CudaMatrix<Type>::set(const int row, const int col, const Type value)
{
	if (row < 0 || row >= rows || col < 0 || col >= cols)
		throw out_of_range("Index out of range.");
	CUDA_CHECK(cudaMemcpy(mat + row * cols + col, &value, sizeof(Type), cudaMemcpyHostToDevice));
}

template<typename Type>
Type* CudaMatrix<Type>::data() const { return this->mat; }

template<typename Type>
void CudaMatrix<Type>::print()
{
	Type* host_data = new Type[rows * cols];
	CUDA_CHECK(cudaMemcpy(host_data, mat, rows * cols * sizeof(Type), cudaMemcpyDeviceToHost));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			cout << host_data[i * cols + j] << " ";
		cout << endl;
	}
	delete[] host_data;
}

template<typename Type>
int CudaMatrix<Type>::getRows() const { return rows; }

template<typename Type>
int CudaMatrix<Type>::getCols() const { return cols; }

template<typename Type>
void CudaMatrix<Type>::getData(Type* dst) const { cudaMemcpy(dst, mat, static_cast<size_t>(rows) * cols * sizeof(Type), cudaMemcpyDeviceToHost); }

template<typename Type>
void CudaMatrix<Type>::setData(const vector<Type>& src) { cudaMemcpy(mat, src.data(), src.size(), cudaMemcpyHostToDevice); }

template<typename Type>
Type CudaMatrix<Type>::get(const int row, const int col) const
{
	if (row < 0 || row >= rows || col < 0 || col >= cols)
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
	elementwise_add_kernel<Type, T, Type> << <gridSize, blockSize >> > (mat, other.mat, mat, total_elements);
}