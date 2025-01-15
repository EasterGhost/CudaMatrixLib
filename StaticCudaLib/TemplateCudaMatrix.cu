#include "TemplateCudaMatrix.cuh"

template <typename Type>
__global__ static void ones_matrix_kernel(Type* data, int total_elements)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		data[idx] = 1;
}

template <typename Type>
__global__ static void col_vec_broadcast_kernel
(const Type* src_vec, Type* res, int size, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		int row = idx / cols;
		res[idx] = src_vec[row];
	}
}

template <typename Type>
__global__ static void row_vec_broadcast_kernel
(const Type* src_vec, Type* res, int size, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		int col = idx % cols;
		res[idx] = src_vec[col];
	}
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_multiply_kernel
(const T1* src1, const T2* src2, T3* res, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = src1[idx] * src2[idx];
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_add_kernel
(const T1* src1, const T2* src2, T3* res, int size, float alpha, float beta)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = alpha * src1[idx] + beta * src2[idx];
}

template <typename T1, typename T2, typename T3>
__global__ static void elementwise_divide_kernel
(const T1* src1, const T2* src2, T3* res, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		res[idx] = src1[idx] / src2[idx];
}

template <typename Type>
__global__ static void identity_matrix_kernel(Type* data, int rows)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < rows)
		data[idx * rows + idx] = 1;
}

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
static dim3 autoSetBlockSize2D(T func, int rows, int cols)
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
}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(int rows, int cols, MatrixType type) : rows(rows), cols(cols)
{
	int total_elements = rows * cols;
	cudaMalloc((void**)&mat, total_elements * sizeof(Type));
	int blockSize = 0;
	int gridSize = 0;

	switch (type)
	{
	case Zero:
		cudaMemset(mat, 0, total_elements * sizeof(Type));
		break;
	case Ones:
		blockSize = autoSetBlockSize(ones);
		gridSize = (total_elements + blockSize - 1) / blockSize;
		ones_matrix_kernel<Type> << <gridSize, blockSize >> > (mat, total_elements);
		break;
	case Identity:
		if (rows != cols)
		{
			throw runtime_error("Identity matrix must be square matrix.");
		}
		cudaMemset(mat, 0, static_cast<size_t>(rows) * cols * sizeof(Type));
		blockSize = autoSetBlockSize(identity);
		gridSize = (rows + blockSize - 1) / blockSize;
		identity_matrix_kernel<Type> << <gridSize, blockSize >> > (mat, rows);
		break;
	case Random:
		curandGenerator_t gen;
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
		curandGenerateUniform(gen, mat, rows * cols);
		curandDestroyGenerator(gen);
		break;
	default:
		throw runtime_error("Unknown matrix type.");
	}
}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(int size) : CudaMatrix(size, size) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(int size, MatrixType type) : CudaMatrix(size, size, type) {}

template <typename Type>
CudaMatrix<Type>::CudaMatrix(int rows, int cols, Type* src) : rows(rows), cols(cols)
{
	int total_elements = rows * cols;
	cudaMalloc((void**)mat, total_elements * sizeof(Type));
	cudaMemcpy(mat, src, total_elements * sizeof(Type), cudaMemcpyHostToDevice);
}

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
}

template <typename Type>
CudaMatrix<Type>::~CudaMatrix()
{
	cudaMemset(mat, 0, static_cast<size_t>(rows) * cols * sizeof(Type));
	rows = 0;
	cols = 0;
	cudaFree(mat);
	mat = nullptr;
}

template<typename Type>
void CudaMatrix<Type>::set(const int row, const int col, const Type value)
{
	if (row < 0 || row >= rows || col < 0 || col >= cols)
		throw out_of_range("Ë÷Òý³¬³ö·¶Î§¡£");
	cudaMemcpy(mat + row * cols + col, &value, sizeof(Type), cudaMemcpyHostToDevice);
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
Type* CudaMatrix<Type>::getData() const
{
	Type* host_data = new Type[rows * cols];
	cudaMemcpy(host_data, mat, static_cast<size_t>(rows) * cols * sizeof(Type), cudaMemcpyDeviceToHost);
	return host_data;
}

template<typename Type>
void CudaMatrix<Type>::setData(const vector<Type>& src) { cudaMemcpy(mat, src.data(), src.size(), cudaMemcpyHostToDevice); }

template<typename Type>
Type CudaMatrix<Type>::get(const int row, const int col) const
{
	if (row < 0 || row >= rows || col < 0 || col >= cols)
		throw out_of_range("Ë÷Òý³¬³ö·¶Î§¡£");
	Type res = 0;
	cudaMemcpy(&res, mat + row * cols + col, sizeof(Type), cudaMemcpyDeviceToHost);
	return res;
}

template<typename Type>
template<typename T>
void CudaMatrix<Type>::add(const CudaMatrix<T>& other)
{
	if (rows != other.rows || cols != other.cols)
		throw runtime_error("¾ØÕóÎ¬¶È²»Æ¥Åä¡£");
	int total_elements = rows * cols;
	int blockSize = autoSetBlockSize(elementwise_add_kernel<Type, T, Type>);
	int gridSize = (total_elements + blockSize - 1) / blockSize;
	elementwise_add_kernel<Type, T, Type> << <gridSize, blockSize >> > (mat, other.mat, mat, total_elements, 1, 1);
}