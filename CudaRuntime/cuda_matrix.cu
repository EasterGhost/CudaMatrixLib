/**
* @file cuda_matrix.cu
* @brief CUDA 矩阵类实现文件
* @note 仅支持 float 类型
* @note 使用 cuBLAS 实现矩阵运算
* @date 2024-12-16
* @version 1.0
* @author LiMuchen
* @license MIT
*/
#include "cuda_matrix.h"

template<typename data_type>
__global__ static void reshape_kernel(const data_type* data, data_type* result, int rows_old, int cols_old, int rows_new, int cols_new) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idx < rows_new && idy < cols_new && idx < rows_old && idy < cols_old) {
		result[idx * cols_new + idy] = data[idx * cols_old + idy];
	}
}

template<typename data_type>
__global__ static void fill_diag_kernel(const data_type* matrix, data_type* diag, int offset, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		int col = idx + offset;
		int row = idx;
		if (col < size && col >= 0)
			matrix[row * size + col] = diag[idx];
	}
}

template<typename data_type>
__device__ static double atomicAdd_double(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}

template<typename data_type>
__global__ static void get_diag(const data_type* matrix, data_type* result, const int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		result[idx] = (data_type)matrix[idx * size + idx];
	}
}

template<typename data_type>
__global__ static void reduce_sum(const data_type* d_input, data_type* d_output, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	float temp = 0.0;
	// 使用归约方法进行求和
	while (tid < n) {
		temp += d_input[tid];
		tid += stride;
	}

	// 将每个线程的结果存到共享内存
	extern __shared__ data_type shared_sum[512];
	int lane = threadIdx.x;

	shared_sum[lane] = temp;
	__syncthreads();

	// 归约：每个block内部合并
	for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
		if (lane < offset) {
			shared_sum[lane] += shared_sum[lane + offset];
		}
		__syncthreads();
	}

	// 最后一个线程将结果写到全局内存
	if (lane == 0) {
		atomicAdd(d_output, shared_sum[0]);
	}
}

template<typename data_type>
__global__ static void reduce_multi(const data_type* d_input, data_type* d_output, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	float temp = 1.0f;
	// 使用归约方法进行求和
	while (tid < n) {
		temp *= d_input[tid];
		tid += stride;
	}

	// 将每个线程的结果存到共享内存
	extern __shared__ data_type shared_sum[512];
	int lane = threadIdx.x;

	shared_sum[lane] = temp;
	__syncthreads();

	// 归约：每个block内部合并
	for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
		if (lane < offset) {
			shared_sum[lane] *= shared_sum[lane + offset];
		}
		__syncthreads();
	}

	// 最后一个线程将结果写到全局内存
	if (lane == 0) {
		atomicAdd(d_output, shared_sum[0]);
	}
}

template<typename data_type>
__global__ static void norm_kernel(const data_type* matrix, data_type* result, const int size, const int L) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		result[idx] = pow(abs(matrix[idx]), L);
	}
}

template<typename data_type>
__global__ static void divide_kernel(data_type* A, data_type* B, data_type* C, const int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		C[idx] = A[idx] / B[idx];
	}
}

__global__ static void random_kernel(curandState* state, unsigned long seed) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, idx, 0, &state[idx]);
}

template<typename data_type>
__global__ static void generate_random_numbers(curandState* globalState, data_type* data, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		curandState localState = globalState[idx];
		data[idx] = curand_uniform(&localState);
		globalState[idx] = localState;
	}
}

template<typename data_type>
__global__ static void identity_matrix_kernel(data_type* data, int rows, int cols) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int total_elements = rows * cols;
	if (idx < total_elements) {
		int row = idx / cols;
		int col = idx % cols;
		if (row == col) { data[idx] = (data_type)1; }
	}
}

template<typename data_type>
__global__ static void ones_matrix_kernel(data_type* data, int total_elements) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_elements) {
		data[idx] = (data_type)1;
	}
}

template<typename data_type>
__global__ static void elementwise_multiply_kernel(const data_type* A, const data_type* B, data_type* C, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		C[idx] = A[idx] * B[idx];
	}
}

template<typename data_type>
__global__ static void col_vec_broadcast2matrix_kernel(const data_type* sourceVector, data_type* resultMatrix, int cols, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		int row = idx / cols;
		resultMatrix[idx] = sourceVector[row];
	}
}

template<typename data_type>
__global__ static void row_vec_broadcast2matrix_kernel(const data_type* sourceVector, data_type* resultMatrix, int cols, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		int col = idx % cols;
		resultMatrix[idx] = sourceVector[col];
	}
}

template<typename data_type>
__global__ static void setdiff_kernel(const data_type* A, const data_type* B, data_type* C, int sizeA, int sizeB, int* count) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < sizeA) {
		bool found = false;
		for (int j = 0; j < sizeB; ++j) {
			if (A[idx] == B[j]) {
				found = true;
				break;
			}
		}
		if (!found) {
			int pos = atomicAdd(count, 1);
			C[pos] = A[idx];
		}
	}
}

static void checkCudaError(cudaError_t err, const char* msg) {
	if (err != cudaSuccess) {
		std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
		system("pause");
		exit(EXIT_FAILURE);
	}
}

template<typename T>
static int autoSetBlockSize(T func) {
	int blockSize = 0;
	int minGridSize = 0;
	cudaError_t status = cudaOccupancyMaxPotentialBlockSize(
		&minGridSize,	// 返回的最小网格大小
		&blockSize,		// 返回的最佳线程块大小
		func,			// 内核函数
		0,				// 动态共享内存大小
		0);				// 块大小限制
	if (status != cudaSuccess) {
		throw runtime_error("cudaOccupancyMaxPotentialBlockSize failed!" + string(cudaGetErrorString(status)));
	}
	return blockSize;
}

template<typename T>
static dim3 autoSetBlockSize2D(T func, int rows, int cols) {
	int blockSize = 0;
	int minGridSize = 0;
	cudaError_t status = cudaOccupancyMaxPotentialBlockSize(
		&minGridSize,  // 返回的最小网格大小
		&blockSize,    // 返回的最佳线程块大小
		func,          // 内核函数
		0,             // 动态共享内存大小
		0);            // 块大小限制
	if (status != cudaSuccess) {
		throw runtime_error("cudaOccupancyMaxPotentialBlockSize failed!" + string(cudaGetErrorString(status)));
	}

	// 计算二维线程块的大小
	int blockDimX = sqrt(blockSize);
	int blockDimY = blockSize / blockDimX;

	// 确保线程块大小不超过矩阵的维度
	blockDimX = min(blockDimX, rows);
	blockDimY = min(blockDimY, cols);

	return dim3(blockDimX, blockDimY);
}
template<typename data_type>
static void findMax(const vector<data_type>& data, int start, int end, data_type& max_value, mutex& mtx) {
	data_type local_max = -numeric_limits<data_type>::infinity();
	for (int i = start; i < end; ++i) {
		if (data[i] > local_max) {
			local_max = data[i];
		}
	}
	// 使用互斥锁保护对全局最大值的更新
	std::lock_guard<mutex> lock(mtx);
	if (local_max > max_value) {
		max_value = local_max;
	}
}

template<typename data_type>
cudaMatrix<data_type>::cudaMatrix() : rows(0), cols(0), data(nullptr) {}

template<typename data_type>
cudaMatrix<data_type>::cudaMatrix(int rows, int cols) : rows(rows), cols(cols) {
	cudaError_t err = cudaMalloc((void**)&data,
		static_cast<size_t>(rows) * cols * sizeof(data_type));
	if (err != cudaSuccess) {
		throw runtime_error("cudaMalloc failed! (code M0001)"
			+ string(cudaGetErrorString(err)));
	}
	cudaMemset(&data, 0, static_cast<size_t>(rows) * cols * sizeof(data_type));
}

template<typename data_type>
cudaMatrix<data_type>::cudaMatrix(int size) : rows(size), cols(size) {
	cudaError_t err = cudaMalloc((void**)&data,
		static_cast<size_t>(rows) * cols * sizeof(data_type));
	if (err != cudaSuccess) {
		throw runtime_error("cudaMalloc failed! (code M0002)"
			+ string(cudaGetErrorString(err)));
	}
	cudaMemset(data, 0, static_cast<size_t>(rows) * cols * sizeof(data_type));
}

template<typename data_type>
cudaMatrix<data_type>::cudaMatrix(int rows, int cols, MatrixType type) : rows(rows), cols(cols) {
	int size = rows * cols;
	cudaError_t err = cudaMalloc((void**)&data, size * sizeof(data_type));
	if (err != cudaSuccess) {
		throw runtime_error("cudaMalloc failed! (code M0003)"
			+ string(cudaGetErrorString(err)));
	}
	int threadsPerBlock = 0;
	int blocksPerGrid = 0;
	switch (type)
	{
	case Zero:
		cudaMemset(data, 0, size * sizeof(data_type));
		break;
	case Ones:
		threadsPerBlock = autoSetBlockSize(ones_matrix_kernel<data_type>);
		blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		ones_matrix_kernel<data_type> << <blocksPerGrid, threadsPerBlock >> > (data, size);
		break;
	case Identity:
		cudaMemset(data, 0, size * sizeof(data_type));
		threadsPerBlock = autoSetBlockSize(identity_matrix_kernel<data_type>);
		blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		identity_matrix_kernel<data_type> << <blocksPerGrid, threadsPerBlock >> > (data, rows, cols);
		break;
	case Random:
		curandState* state = nullptr;
		cudaError_t err = cudaMalloc((void**)&state, size * sizeof(curandState));
		if (err != cudaSuccess) {
			throw runtime_error("cudaMalloc failed for curandState! (code M0005)"
				+ string(cudaGetErrorString(err)));
		}
		threadsPerBlock = autoSetBlockSize(random_kernel);
		blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		random_kernel << <blocksPerGrid, threadsPerBlock >> > (state, (unsigned long)time(0));
		generate_random_numbers<data_type> << <blocksPerGrid, threadsPerBlock >> > (state, data, size);
		cudaFree(state);
		break;
	}
}

template<typename data_type>
cudaMatrix<data_type>::cudaMatrix(int size, MatrixType type) : rows(size), cols(size) {
	int total_elements = rows * cols;
	cudaError_t err = cudaMalloc((void**)&data, total_elements * sizeof(data_type));
	if (err != cudaSuccess) {
		throw runtime_error("cudaMalloc failed! (code M0004)"
			+ string(cudaGetErrorString(err)));
	}
	int threadsPerBlock = 0;
	int blocksPerGrid = 0;
	switch (type)
	{
	case Zero:
		cudaMemset(data, 0, total_elements * sizeof(data_type));
		break;
	case Ones:
		threadsPerBlock = autoSetBlockSize(ones_matrix_kernel<data_type>);
		blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
		ones_matrix_kernel<data_type> << <blocksPerGrid, threadsPerBlock >> > (data, total_elements);
		break;
	case Identity:
		cudaMemset(data, 0, total_elements * sizeof(data_type));
		threadsPerBlock = autoSetBlockSize(identity_matrix_kernel<data_type>);
		blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
		identity_matrix_kernel<data_type> << <blocksPerGrid, threadsPerBlock >> > (data, rows, cols);
		break;
	case Random:
		curandState* state = nullptr;
		cudaError_t err = cudaMalloc((void**)&state, total_elements * sizeof(curandState));
		if (err != cudaSuccess) {
			throw runtime_error("cudaMalloc failed for curandState! (code M0005)");
		}
		threadsPerBlock = autoSetBlockSize(random_kernel);
		blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
		random_kernel << <blocksPerGrid, threadsPerBlock >> > (state, (unsigned long)time(0));
		generate_random_numbers<data_type> << <blocksPerGrid, threadsPerBlock >> > (state, data, total_elements);
		cudaFree(state);
		break;
	}
}

template<typename data_type>
cudaMatrix<data_type>::cudaMatrix(const cudaMatrix<data_type>& others) : rows(others.rows), cols(others.cols) {
	cudaError_t err = cudaMalloc((void**)&this->data, static_cast<size_t>(rows) * cols * sizeof(data_type));
	if (err != cudaSuccess) {
		throw runtime_error("cudaMalloc failed! (code M0006)");
	}
	cudaMemcpy(this->data, others.data, static_cast<size_t>(rows) * cols * sizeof(data_type), cudaMemcpyDeviceToDevice);
}

template<typename data_type>
cudaMatrix<data_type>::~cudaMatrix() {
	cudaMemset(data, 0, static_cast<size_t>(rows) * cols * sizeof(data_type));
	cudaFree(data);
	rows = 0;
	cols = 0;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::fromValue(data_type value) {
	cudaMatrix result(1);
	cudaMemcpy(result.data, &value, sizeof(data_type), cudaMemcpyHostToDevice);
	return result;
}

template<typename data_type>
void cudaMatrix<data_type>::resize(int rows, int cols) {
	if (this->rows == rows && this->cols == cols) { return; }
	float* new_data = nullptr;
	cudaMalloc((void**)&new_data, static_cast<size_t>(rows) * cols * sizeof(data_type));
	cudaMemset(new_data, 0, static_cast<size_t>(rows) * cols * sizeof(data_type));
	dim3 threadsPerBlock = autoSetBlockSize2D(reshape_kernel<data_type>, rows, cols);
	dim3 blocksPerGrid = dim3((rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);
	reshape_kernel << <blocksPerGrid, threadsPerBlock >> > (data, new_data, this->rows, this->cols, rows, cols);
	cudaFree(data);
	data = new_data;
	this->rows = rows;
	this->cols = cols;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::zeros(int rows, int cols) { return cudaMatrix(rows, cols); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::zeros(int size) { return cudaMatrix(size); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::ones(int rows, int cols) { return cudaMatrix(rows, cols, Ones); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::ones(int size) { return cudaMatrix(size, Ones); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::identity(int size) { return cudaMatrix(size, Identity); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::random(int rows, int cols) { return cudaMatrix(rows, cols, Random); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::random(int size) { return cudaMatrix(size, Random); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator=(const cudaMatrix<data_type>& B) {
	if (this == &B) { return *this; }
	if (rows != B.rows || cols != B.cols) {
		cudaFree(data);
		cudaMalloc((void**)&data, static_cast<size_t>(B.rows) * B.cols * sizeof(data_type));
		rows = B.rows;
		cols = B.cols;
	}
	cudaMemcpy(data, B.data, static_cast<size_t>(B.rows) * B.cols * sizeof(data_type), cudaMemcpyDeviceToDevice);
	return *this;
}

template<typename data_type>
void cudaMatrix<data_type>::set(int row, int col, data_type value) {
	if (row < 0 || row >= rows || col < 0 || col >= cols) {
		throw out_of_range("索引超出范围。");
	}
	cudaMemcpy(data + static_cast<size_t>(row) * cols + col, &value, sizeof(data_type), cudaMemcpyHostToDevice);
}

template<typename data_type>
void cudaMatrix<data_type>::setData(const vector<data_type>& v) { cudaMemcpy(data, v.data(), v.size() * sizeof(data_type), cudaMemcpyHostToDevice); }

template<typename data_type>
data_type cudaMatrix<data_type>::get(int row, int col) const {
	if (row < 0 || row >= rows || col < 0 || col >= cols) {
		throw out_of_range("索引超出范围。");
	}
	data_type result = 0;
	cudaMemcpy(&result, data + static_cast<size_t>(row) * cols + col,
		sizeof(data_type), cudaMemcpyDeviceToHost);
	return result;
}

template<typename data_type>
void cudaMatrix<data_type>::getData(vector<data_type>& v, ...) const {
	va_list args;
	va_start(args, &v);
	if (va_arg(args, bool)) { // 如果第一个参数为 true，则强制重新分配内存
		v.resize(static_cast<size_t>(rows) * cols);
	}
	va_end(args);
	cudaMemcpy(v.data(), data,
		v.size() * sizeof(data_type), cudaMemcpyDeviceToHost);
}

template<typename data_type>
data_type* cudaMatrix<data_type>::getDataPtr() const { return data; }

template<typename data_type>
int cudaMatrix<data_type>::getRows() const { return rows; }

template<typename data_type>
int cudaMatrix<data_type>::getCols() const { return cols; }

template<typename data_type>
void cudaMatrix<data_type>::printData() const {
	vector<data_type> hostData(rows * cols);
	cudaMemcpy(hostData.data(), data,
		static_cast<size_t>(rows) * cols * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			cout << hostData[static_cast<size_t>(i) * cols + j] << " ";
		}
		cout << endl;
	}
}

template<typename data_type>
float cudaMatrix<data_type>::norm(int L) const {
	if (rows != 1 && cols != 1)
		throw invalid_argument("输入不是向量，无法求范数。");
	if (L <= 0)
		throw invalid_argument("范数阶数必须大于 0。");
	int size = max(rows, cols);
	int threadsPerBlock = autoSetBlockSize(norm_kernel<data_type>);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	data_type* vec2 = nullptr;
	cudaMalloc((void**)&vec2, size * sizeof(data_type));
	norm_kernel << <blocksPerGrid, threadsPerBlock >> > (data, vec2, size, L);
	data_type* result = nullptr;
	cudaMalloc((void**)&result, sizeof(data_type));
	cudaMemset(result, 0, sizeof(data_type));
	reduce_sum<data_type> << <1, 512, 512 * sizeof(data_type) >> > (vec2, result, size);
	//cudaDeviceSynchronize();
	data_type result_host = (data_type)0;
	cudaMemcpy(&result_host, result, sizeof(data_type), cudaMemcpyDeviceToHost);
	cudaFree(result);
	return powf(result_host, 1.0f / L);
}

template<typename data_type>
cudaMatrix<data_type>::operator data_type() const {
	if (rows != 1 || cols != 1)
		throw invalid_argument("矩阵规模不是 1x1，无法转换为标量。");
	data_type result = (data_type)0;
	cudaMemcpy(&result, data, sizeof(data_type), cudaMemcpyDeviceToHost);
	return result;
}

template<typename data_type>
bool cudaMatrix<data_type>::operator<(const data_type n) {
	if (rows != 1 || cols != 1)
		throw invalid_argument("输入不是标量，无法比较。");
	return data[0] < n;
}

template<typename data_type>
bool cudaMatrix<data_type>::operator<(const cudaMatrix<data_type>& B) {
	if (rows != 1 || cols != 1 || B.rows != 1 || B.cols != 1)
		throw invalid_argument("输入不是标量，无法比较。");
	return data[0] < B.data[0];
}

template<typename data_type>
bool cudaMatrix<data_type>::operator<=(const data_type n) {
	if (rows != 1 || cols != 1)
		throw invalid_argument("输入不是标量，无法比较。");
	return data[0] <= n;
}

template<typename data_type>
bool cudaMatrix<data_type>::operator<=(const cudaMatrix<data_type>& B) {
	if (rows != 1 || cols != 1 || B.rows != 1 || B.cols != 1)
		throw invalid_argument("输入不是标量，无法比较。");
	return data[0] <= B.data[0];
}

template<typename data_type>
bool cudaMatrix<data_type>::operator>(const data_type n) {
	if (rows != 1 || cols != 1)
		throw invalid_argument("输入不是标量，无法比较。");
	return data[0] > n;
}

template<typename data_type>
bool cudaMatrix<data_type>::operator>(const cudaMatrix<data_type>& B) {
	if (rows != 1 || cols != 1 || B.rows != 1 || B.cols != 1)
		throw invalid_argument("输入不是标量，无法比较。");
	return data[0] > B.data[0];
}

template<typename data_type>
bool cudaMatrix<data_type>::operator>=(const data_type n) {
	if (rows != 1 || cols != 1)
		throw invalid_argument("输入不是标量，无法比较。");
	return data[0] >= n;
}

template<typename data_type>
bool cudaMatrix<data_type>::operator>=(const cudaMatrix<data_type>& B) {
	if (rows != 1 || cols != 1 || B.rows != 1 || B.cols != 1)
		throw invalid_argument("输入不是标量，无法比较。");
	return data[0] >= B.data[0];
}

template<typename data_type>
void cudaMatrix<data_type>::add(const cudaMatrix& B) {
	if (rows != B.rows || cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法相加。");
	}
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	const float alpha = 1.0f;
	const float beta = 1.0f;
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		rows, cols,
		&alpha, data, rows,
		&beta, B.data, B.rows,
		data, rows);
	cublasDestroy_v2(handle);
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::add(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B) {
	if (A.rows != B.rows || A.cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法相加。");
	}
	cudaMatrix<data_type> result(A.rows, A.cols);
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	const float alpha = 1.0f;
	const float beta = 1.0f;

	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		A.rows, A.cols,
		&alpha, A.data, A.rows,
		&beta, B.data, B.rows,
		result.data, result.rows);

	cublasDestroy_v2(handle);
	return result;
}

template<typename data_type>
cudaMatrix< data_type> cudaMatrix<data_type>::operator+(const cudaMatrix<data_type>& B) { return add(*this, B); }

template<typename data_type>
cudaMatrix< data_type> cudaMatrix<data_type>::operator+=(const cudaMatrix<data_type>& B) {
	this->add(const_cast<cudaMatrix<data_type>&>(B));
	return *this;
}

template<typename data_type>
void cudaMatrix<data_type>::subtract(const cudaMatrix<data_type>& B) {
	if (rows != B.rows || cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法相减。");
	}
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	const float alpha = 1.0f;
	const float beta = -1.0f;
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		rows, cols,
		&alpha, data, rows,
		&beta, B.data, B.rows,
		data, rows);
	cublasDestroy_v2(handle);
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::subtract(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B) {
	if (A.rows != B.rows || A.cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法相减。");
	}
	cudaMatrix<data_type> result(A.rows, A.cols);
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	const float alpha = 1.0f;
	const float beta = -1.0f;

	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		A.rows, A.cols,
		&alpha, A.data, A.rows,
		&beta, B.data, B.rows,
		result.data, result.rows);

	cublasDestroy_v2(handle);
	return result;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator-(const cudaMatrix& B) { return subtract(*this, B); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator-=(const cudaMatrix& B) {
	this->subtract(const_cast<cudaMatrix<data_type>&>(B));
	return *this;
}

template<typename data_type>
void cudaMatrix<data_type>::multiply(cudaMatrix<data_type>& B) { // 请不要管这里，这里是屎山
	if (rows == 1 && cols == 1) {
		cublasHandle_t handle;
		cublasCreate_v2(&handle);
		cublasSscal_v2(handle, rows * cols, data, B.data, 1);
		cublasDestroy_v2(handle);
		return;
	}
	if (cols != B.rows) {
		throw invalid_argument("矩阵维度不匹配，无法相乘。");
	}
	cudaMatrix<data_type> temp(rows, B.cols);
	cudaMatrix<data_type> tempA = transpose();
	cudaMatrix<data_type> tempB = B.transpose();
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		rows, B.cols, cols,
		&alpha, tempA.data, tempA.cols,
		tempB.data, tempB.cols,
		&beta, temp.data, temp.rows);
	int tmp = temp.cols;
	temp.cols = temp.rows;
	temp.rows = tmp;
	cudaMemcpy(data, temp.transpose().data, static_cast<size_t>(rows) * cols * sizeof(float), cudaMemcpyDeviceToDevice);
	cublasDestroy_v2(handle);
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::multiply(const cudaMatrix& A, const cudaMatrix& B) { // 请不要管这里，这里是屎山
	if (A.rows == 1 && A.cols == 1) {
		cudaMatrix<data_type> result(B);
		cublasHandle_t handle;
		cublasCreate_v2(&handle);
		cublasSscal_v2(handle, B.rows * B.cols, A.data, result.data, 1);
		cublasDestroy_v2(handle);
		return result;
	}
	if (A.cols != B.rows) {
		throw invalid_argument("矩阵维度不匹配，无法相乘。");
	}
	cudaMatrix<data_type> temp(A.rows, B.cols);
	cudaMatrix<data_type> tempA = A.transpose();
	cudaMatrix<data_type> tempB = B.transpose();
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	const float alpha = 1.0f;
	const float beta = 0.0f;

	cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		A.rows, B.cols, A.cols,
		&alpha, tempA.data, tempA.cols,
		tempB.data, tempB.cols,
		&beta, temp.data, temp.rows);
	int tmp = temp.cols;
	temp.cols = temp.rows;
	temp.rows = tmp;
	cudaMatrix<data_type> result = temp.transpose();
	cublasDestroy_v2(handle);
	return result;
}

template<typename data_type>
cudaMatrix<data_type> operator * (const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B) { return cudaMatrix<data_type>::multiply(A, B); }

template<typename data_type>
cudaMatrix<data_type> operator *(const data_type scalar, const cudaMatrix<data_type>& A) { return A.scalarMultiply(scalar); }

template<typename data_type>
cudaMatrix<data_type> operator *(const cudaMatrix<data_type>& A, const data_type scalar) { return A.scalarMultiply(scalar); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator *= (const cudaMatrix& B) {
	this->multiply(const_cast<cudaMatrix<data_type>&>(B));
	return *this;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator *= (const data_type scalar) {
	this->scalarMultiply(scalar);
	return *this;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator ^ (int pows) {
	if (rows != cols) {
		throw invalid_argument("矩阵不是方阵，无法求幂。");
	}
	if (pows < 0) {
		throw invalid_argument("幂次必须大于等于 0。");
	}
	cudaMatrix<data_type> result(rows, cols, Identity);
	cudaMatrix<data_type> base = *this;
	while (pows > 0) {
		if (pows % 2 == 1) {
			result.multiply(base);
		}
		base = multiply(base, base);
		pows /= 2;
	}
	return result;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator ^= (int pows) {
	if (rows != cols) {
		throw invalid_argument("矩阵不是方阵，无法求幂。");
	}
	if (pows < 0) {
		throw invalid_argument("幂次必须大于等于 0。");
	}
	pows--;
	cudaMatrix<data_type> base = *this;
	while (pows > 0) {
		if (pows % 2 == 1) {
			*this = multiply(*this, base);
		}
		base = multiply(base, base);
		pows /= 2;
	}
	return *this;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::transpose() const {
	cudaMatrix<data_type> result(cols, rows);
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	const float alpha = 1.0f;
	const float beta = 0.0f;

	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
		rows, cols,
		&alpha, data, cols,
		&beta, data, rows,
		result.data, result.cols);

	cublasDestroy_v2(handle);
	return result;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::transpose(const cudaMatrix& A) { return A.transpose(); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator ~ () const { return this->transpose(); }

template<typename data_type>
data_type cudaMatrix<data_type>::trace() const {
	if (rows != cols) {
		throw invalid_argument("矩阵不是方阵，无法求迹。");
	}
	int size = rows;
	data_type* trace_array = nullptr;
	cudaMalloc((void**)&trace_array, size * sizeof(data_type));
	int threadsPerBlock = autoSetBlockSize(get_diag<data_type>);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	get_diag<data_type> << <blocksPerGrid, threadsPerBlock >> > (data, trace_array, size);
	cudaDeviceSynchronize();
	data_type* d_result = nullptr;
	cudaMalloc((void**)&d_result, sizeof(data_type));
	cudaMemset(d_result, 0, sizeof(data_type));
	reduce_sum<data_type> << <1, 512, 512 * sizeof(data_type) >> > (trace_array, d_result, size);
	data_type result = 0;
	cudaMemcpy(&result, d_result, sizeof(data_type), cudaMemcpyDeviceToHost);
	cudaFree(trace_array);
	cudaFree(d_result);
	return result;
}

template<typename data_type>
data_type cudaMatrix<data_type>::trace(const cudaMatrix<data_type>& A) { return A.trace(); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::scalarMultiply(data_type scalar) const {
	cudaMatrix<data_type> result(rows, cols);
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	cudaMemcpy(result.data, data, static_cast<size_t>(rows) * cols * sizeof(data_type), cudaMemcpyDeviceToDevice);
	cublasSscal_v2(handle, rows * cols, &scalar, result.data, 1);
	cublasDestroy_v2(handle);
	return result;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::matrixDOTmatrix(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B) {
	if (A.rows != B.rows || A.cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法进行点乘。");
	}
	cudaMatrix<data_type> result(A.rows, A.cols);
	int size = A.rows * A.cols;
	int threadsPerBlock = autoSetBlockSize(elementwise_multiply_kernel<data_type>);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	elementwise_multiply_kernel<data_type> << <blocksPerGrid, threadsPerBlock >> > (A.data, B.data, result.data, size);
	return result;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::vectorBroadcast2Matrix(const cudaMatrix<data_type>& sourceVector, const int rows, const int cols) {
	if (sourceVector.cols != 1 && sourceVector.rows != 1) {
		throw invalid_argument("输入矩阵不是向量，无法进行广播。");
	}
	if (rows <= 0 || cols <= 0) {
		throw invalid_argument("广播长度必须大于 0。");
	}
	int size = rows * cols;
	cudaMatrix<data_type> result(rows, cols);
	if (sourceVector.cols == 1) {
		if (sourceVector.rows != rows)
			throw invalid_argument("向量长度与广播长度不匹配。");
		int threadsPerBlock = autoSetBlockSize(col_vec_broadcast2matrix_kernel<data_type>);
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		col_vec_broadcast2matrix_kernel<data_type> << <blocksPerGrid, threadsPerBlock >> > (sourceVector.data, result.data, cols, size);
	}
	else if (sourceVector.rows == 1) {
		if (sourceVector.cols != cols)
			throw invalid_argument("向量长度与广播长度不匹配。");
		int threadsPerBlock = autoSetBlockSize(row_vec_broadcast2matrix_kernel<data_type>);
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		row_vec_broadcast2matrix_kernel<data_type> << <blocksPerGrid, threadsPerBlock >> > (sourceVector.data, result.data, cols, size);
	}
	return result;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::dot(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B) {
	if (A.rows == B.rows && A.cols == B.cols) {
		return matrixDOTmatrix(A, B);
	}
	int row = A.rows > B.rows ? A.rows : B.rows;
	int col = A.cols > B.cols ? A.cols : B.cols;
	cudaMatrix tempA(row, col);
	cudaMatrix tempB(row, col);
	if (A.cols == 1 || A.rows == 1) {
		cudaMatrix temp1 = vectorBroadcast2Matrix(A, row, col);
		cudaMemcpy(tempA.data, temp1.data, static_cast<size_t>(row) * col * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	else
		cudaMemcpy(tempA.data, A.data, static_cast<size_t>(A.rows) * A.cols * sizeof(float), cudaMemcpyDeviceToDevice);
	if (B.cols == 1 || B.rows == 1) {
		cudaMatrix temp2 = vectorBroadcast2Matrix(B, row, col);
		cudaMemcpy(tempB.data, temp2.data, static_cast<size_t>(row) * col * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	else
		cudaMemcpy(tempB.data, B.data, static_cast<size_t>(row) * col * sizeof(float), cudaMemcpyDeviceToDevice);
	return matrixDOTmatrix(tempA, tempB);
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::dot(const data_type scalar, const cudaMatrix<data_type>& A) { return A.scalarMultiply(scalar); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::dot(const cudaMatrix& A, const data_type scalar) { return A.scalarMultiply(scalar); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::divide(const cudaMatrix& A, const cudaMatrix& B) {
	if (A.rows != B.rows || A.cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法相除。");
	}
	cudaMatrix result(A.rows, A.cols);
	int size = A.rows * A.cols;
	int threadsPerBlock = autoSetBlockSize(divide_kernel<data_type>);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	divide_kernel << <blocksPerGrid, threadsPerBlock >> > (A.data, B.data, result.data, size);
	return result;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator / (const cudaMatrix& B) { return divide(*this, B); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator / (const data_type scalar) {
	float invScalar = 1.0f / scalar;
	cudaMatrix<data_type> result(rows, cols);
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	cudaMemcpy(result.data, data, static_cast<size_t>(rows) * cols * sizeof(data_type), cudaMemcpyDeviceToDevice);
	cublasSscal_v2(handle, rows * cols, &invScalar, result.data, 1);
	cublasDestroy_v2(handle);
	return result;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator /= (const cudaMatrix<data_type>& B) {
	if (rows != B.rows || cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法相除。");
	}
	int size = rows * cols;
	int threadsPerBlock = autoSetBlockSize(divide_kernel<data_type>);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	divide_kernel<data_type> << <blocksPerGrid, threadsPerBlock >> > (data, B.data, data, size);
	return *this;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator /= (const data_type scalar) {
	float invScalar = 1.0f / scalar;
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	cublasSscal_v2(handle, rows * cols, &invScalar, data, 1);
	cublasDestroy_v2(handle);
	return *this;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::solveSparseSLE(cudaMatrix<data_type>& A, cudaMatrix<data_type>& b) {
	if (A.rows != b.rows) {
		throw invalid_argument("矩阵维度不匹配，无法求解稀疏线性方程组。");
	}
	if (b.cols != 1) {
		throw invalid_argument("右侧矩阵不是列向量，无法求解稀疏线性方程组。");
	}
	cudaMatrix<float> x(A.rows, 1, Ones);
	cudaMatrix<float> r = b - A * x;
	cudaMatrix<float> p = r;
	cudaMatrix<float> r_old = r;
	float r_norm2 = ~r * r;
	for (int i = 0; i < 1e6; i++)
	{
		cudaMatrix<float> Ap = A * p;
		float alpha = r_norm2 / (~p * Ap);
		x = alpha * p + x;
		r = r - alpha * Ap;
		float beta = (~r * r) / (~r_old * r_old);
		p = beta * p + r;
		r_old = r;
		r_norm2 = ~r * r;
		if (r_norm2 < 1e-16)
			break;
	}
	return x;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::operator | (cudaMatrix& b) { return solveSparseSLE(*this, b); }

template<typename data_type>
data_type cudaMatrix<data_type>::det() const {
	if (rows != cols) {
		throw invalid_argument("矩阵不是方阵，无法求行列式。");
	}
	int size = rows;
	cusolverDnHandle_t handle;
	cusolverDnCreate(&handle);
	cudaMatrix temp(*this);
	int* Pivots = nullptr;
	int* Info = nullptr;
	cudaMalloc((void**)&Pivots, size * sizeof(int));
	cudaMalloc((void**)&Info, sizeof(int));
	int workspace_size = 0;
	cusolverDnSgetrf_bufferSize(handle, size, size, temp.data, size, &workspace_size);
	float* workspace = nullptr;
	cudaMalloc((void**)&workspace, workspace_size * sizeof(float));
	cusolverDnSgetrf(handle, size, size, temp.data, size, workspace, Pivots, Info);
	float det = 1.0f;
	float* diag = nullptr;
	cudaMalloc((void**)&diag, size * sizeof(float));
	int threadsPerBlock = autoSetBlockSize(get_diag<data_type>);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	get_diag<data_type> << <blocksPerGrid, threadsPerBlock >> > (temp.data, diag, size);
	cudaDeviceSynchronize();
	vector<float> hostDiag(size);
	cudaMemcpy(hostDiag.data(), diag, size * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; ++i) { det *= hostDiag[i]; }
	vector<int> hostPivots(size);
	cudaMemcpy(hostPivots.data(), Pivots, size * sizeof(int), cudaMemcpyDeviceToHost);
	int pivotSign = 1;
	for (int i = 0; i < size; ++i) {
		if (hostPivots[i] != (i + 1))
			pivotSign *= -1;
	}
	det *= pivotSign;
	cudaFree(Pivots);
	cudaFree(Info);
	cudaFree(workspace);
	cusolverDnDestroy(handle);
	return det;
}

template<typename data_type>
data_type cudaMatrix<data_type>::det(const cudaMatrix& A) { return A.det(); }

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::diag(const vector<int> offset, ...) {
	int num = offset.size();
	va_list args;
	va_start(args, offset);
	vector<vector<data_type>> arg(num);
	for (int i = 0; i < num; i++) {
		arg[i] = va_arg(args, vector<data_type>);
		if (arg[i].data() == nullptr)
			throw invalid_argument("输入矩阵指针为空。");
	}
	va_end(args);
	int size = arg[0].size();
	cudaMatrix<data_type> result(size);
	int threadsPerBlock = autoSetBlockSize(fill_diag_kernel<data_type>);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	for (int i = 0; i < num; i++) {
		float* tmp = nullptr;
		cudaMalloc((void**)&tmp, arg[i].size() * sizeof(data_type));
		cudaMemcpy(tmp, arg[i].data(), arg[i].size() * sizeof(data_type), cudaMemcpyHostToDevice);
		fill_diag_kernel<data_type> << <blocksPerGrid, threadsPerBlock >> > (result.data, tmp, offset[i], size);
	}
	return result;
}

template<typename data_type>
cudaMatrix<data_type>::ElementProxy cudaMatrix<data_type>::operator()(int row, int col) {
	if (row < 0 || row >= rows || col < 0 || col >= cols) {
		throw std::out_of_range("索引超出范围。");
	}
	return ElementProxy(*this, row, col);
}

template<typename data_type>
cudaMatrix<data_type>::ElementProxy::~ElementProxy() {
	row = 0;
	col = 0;
	mat = cudaMatrix();
}

template<typename data_type>
cudaMatrix<data_type>::ElementProxy::operator float() const { return mat.get(row, col); }

template<typename data_type>
cudaMatrix<data_type>::ElementProxy& cudaMatrix<data_type>::ElementProxy::operator=(float value) {
	mat.set(row, col, value);
	return *this;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::assembleBlocks(vector<vector<cudaMatrix>>& blocks) {
	if (blocks.empty()) {
		throw std::invalid_argument("Blocks cannot be empty");
	}
	int numBlockRows = blocks.size();
	int numBlockCols = blocks[0].size();
	for (int i = 0; i < numBlockRows; ++i) {
		if (blocks[i].size() != numBlockCols) {
			throw std::invalid_argument("All rows must have the same number of blocks");
		}
	}
	vector<int> maxBlockRows(numBlockRows, 0);
	vector<int> maxBlockCols(numBlockCols, 0);
	for (int i = 0; i < numBlockRows; ++i) {
		for (int j = 0; j < numBlockCols; ++j) {
			int blockRows = blocks[i][j].getRows();
			int blockCols = blocks[i][j].getCols();
			if (blockRows > maxBlockRows[i])
				maxBlockRows[i] = blockRows;
			if (blockCols > maxBlockCols[j])
				maxBlockCols[j] = blockCols;
		}
	}
	for (int i = 0; i < numBlockRows; ++i) {
		for (int j = 0; j < numBlockCols; ++j) {
			blocks[i][j].resize(maxBlockRows[i], maxBlockCols[j]);
		}
	}
	int totalRows = 0;
	for (int i = 0; i < numBlockRows; ++i) {
		totalRows += maxBlockRows[i];
	}
	int totalCols = 0;
	for (int j = 0; j < numBlockCols; ++j)
		totalCols += maxBlockCols[j];
	cudaMatrix<data_type> result(totalRows, totalCols);
	int rowOffset = 0;
	for (int i = 0; i < numBlockRows; ++i) {
		int colOffset = 0;
		for (int j = 0; j < numBlockCols; ++j) {
			const cudaMatrix<data_type>& block = blocks[i][j];
			int blockRows = block.getRows();
			int blockCols = block.getCols();
			const data_type* srcData = block.getDataPtr();
			data_type* destData = result.getDataPtr() + (rowOffset * totalCols + colOffset);
			size_t srcPitch = blockCols * sizeof(float);
			size_t destPitch = totalCols * sizeof(float);
			size_t widthInBytes = blockCols * sizeof(float);
			size_t height = blockRows;
			cudaError_t err = cudaMemcpy2D(destData, destPitch,
				srcData, srcPitch,
				widthInBytes, height,
				cudaMemcpyDeviceToDevice);
			if (err != cudaSuccess) {
				throw runtime_error("cudaMemcpy2D failed: " + string(cudaGetErrorString(err)));
			}
			colOffset += blockCols;
		}
		rowOffset += maxBlockRows[i];
	}
	return result;
}

template<typename data_type>
cudaMatrix<data_type> cudaMatrix<data_type>::setdiff(const cudaMatrix<data_type>& A, const cudaMatrix<data_type>& B) {
	if (A.rows != 1 && A.cols != 1)
		throw invalid_argument("A is not a vector.");
	if (B.rows != 1 && B.cols != 1)
		throw invalid_argument("B is not a vector.");

	int sizeA = max(A.rows, A.cols);
	int sizeB = max(B.rows, B.cols);

	data_type* d_result = nullptr;
	cudaMalloc((void**)&d_result, sizeA * sizeof(data_type));
	int* d_count = nullptr;
	cudaMalloc((void**)&d_count, sizeof(int));
	cudaMemset(d_count, 0, sizeof(int));

	int threadsPerBlock = 256;
	int blocksPerGrid = (sizeA + threadsPerBlock - 1) / threadsPerBlock;
	setdiff_kernel<data_type> << <blocksPerGrid, threadsPerBlock >> > (A.data, B.data, d_result, sizeA, sizeB, d_count);

	int h_count = 0;
	cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

	cudaMatrix result(1, h_count);
	cudaMemcpy(result.data, d_result, h_count * sizeof(float), cudaMemcpyDeviceToDevice);

	cudaFree(d_result);
	cudaFree(d_count);

	return result;
}