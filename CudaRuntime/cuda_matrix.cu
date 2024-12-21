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

extern "C" __global__ static void fill_diag_kernel(float* matrix, float* diag, int offset, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		int col = idx + offset;
		int row = idx;
		if (col < size && col >= 0)
			matrix[row * size + col] = diag[idx];
	}
}

extern "C" __device__ static double atomicAdd_double(double* address, double val)
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

extern "C" __global__ static void get_diag(const float* matrix, float* result, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		result[idx] = (float)matrix[idx * size + idx];
	}
}

extern "C" __global__ static void reduce_sum(float* d_input, float* d_output, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	float temp = 0.0;
	// 使用归约方法进行求和
	while (tid < n) {
		temp += d_input[tid];
		tid += stride;
	}

	// 将每个线程的结果存到共享内存
	extern __shared__ float shared_sum[512];
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

extern "C" __global__ static void reduce_multi(float* d_input, float* d_output, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	float temp = 1.0f;
	// 使用归约方法进行求和
	while (tid < n) {
		temp *= d_input[tid];
		tid += stride;
	}

	// 将每个线程的结果存到共享内存
	extern __shared__ float shared_sum[512];
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

extern "C" __global__ static void norm_kernel(const float* matrix, float* result, int size, int L) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		result[idx] = (float)pow(fabsf(matrix[idx]), L);
	}
}

extern "C" __global__ static void divide_kernel(float* A, float* B, float* C, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		C[idx] = A[idx] / B[idx];
	}
}

extern "C" __global__ static void random_kernel(curandState* state, unsigned long seed) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, idx, 0, &state[idx]);
}

extern "C" __global__ static void generate_random_numbers(curandState* globalState, float* data, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		curandState localState = globalState[idx];
		data[idx] = curand_uniform(&localState);
		globalState[idx] = localState;
	}
}

extern "C" __global__ static void identity_matrix_kernel(float* data, int rows, int cols) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int total_elements = rows * cols;
	if (idx < total_elements) {
		int row = idx / cols;
		int col = idx % cols;
		if (row == col) { data[idx] = 1.0f; }
	}
}

extern "C" __global__ static void ones_matrix_kernel(float* data, int total_elements) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_elements) {
		data[idx] = 1.0f;
	}
}

extern "C" __global__ static void elementwise_multiply_kernel(const float* A, const float* B, float* C, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		C[idx] = A[idx] * B[idx];
	}
}

extern "C" __global__ static void col_vec_broadcast2matrix_kernel(const float* sourceVector, float* resultMatrix, int cols, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		int row = idx / cols;
		resultMatrix[idx] = sourceVector[row];
	}
}

extern "C" __global__ static void row_vec_broadcast2matrix_kernel(const float* sourceVector, float* resultMatrix, int cols, int size) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		int col = idx % cols;
		resultMatrix[idx] = sourceVector[col];
	}
}

static void checkCudaError(cudaError_t err, const char* msg) {
	if (err != cudaSuccess) {
		std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
		system("pause");
		exit(EXIT_FAILURE);
	}
}

template<class T>
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

template<class T>
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

static void findMax(const std::vector<float>& data, int start, int end, float& max_value, std::mutex& mtx) {
	float local_max = -std::numeric_limits<float>::infinity();
	for (int i = start; i < end; ++i) {
		if (data[i] > local_max) {
			local_max = data[i];
		}
	}
	// 使用互斥锁保护对全局最大值的更新
	std::lock_guard<std::mutex> lock(mtx);
	if (local_max > max_value) {
		max_value = local_max;
	}
}

cudaMatrix::cudaMatrix() : rows(0), cols(0), data(nullptr) {}

cudaMatrix::cudaMatrix(int rows, int cols) : rows(rows), cols(cols) {
	cudaError_t err = cudaMalloc((void**)&data, static_cast<size_t>(rows) * cols * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error("cudaMalloc failed! (code M0001)" + string(cudaGetErrorString(err)));
	}
	cudaMemset(&data, 0, static_cast<size_t>(rows) * cols * sizeof(float));
}

cudaMatrix::cudaMatrix(int size) : rows(size), cols(size) {
	cudaError_t err = cudaMalloc((void**)&data, static_cast<size_t>(rows) * cols * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error("cudaMalloc failed! (code M0002)");
	}
	cudaMemset(data, 0, static_cast<size_t>(rows) * cols * sizeof(float));
}

cudaMatrix::cudaMatrix(int rows, int cols, MatrixType type) : rows(rows), cols(cols) {
	int size = rows * cols;
	cudaError_t err = cudaMalloc((void**)&data, size * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error("cudaMalloc failed! (code M0003)");
	}
	int threadsPerBlock = 0;
	int blocksPerGrid = 0;
	switch (type)
	{
	case Zero:
		cudaMemset(data, 0, size * sizeof(float));
		break;
	case Ones:
		threadsPerBlock = autoSetBlockSize(ones_matrix_kernel);
		blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		ones_matrix_kernel << <blocksPerGrid, threadsPerBlock >> > (data, size);
		break;
	case Identity:
		cudaMemset(data, 0, size * sizeof(float));
		threadsPerBlock = autoSetBlockSize(identity_matrix_kernel);
		blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		identity_matrix_kernel << <blocksPerGrid, threadsPerBlock >> > (data, rows, cols);
		break;
	case Random:
		curandState* state = nullptr;
		cudaError_t err = cudaMalloc((void**)&state, size * sizeof(curandState));
		if (err != cudaSuccess) {
			throw runtime_error("cudaMalloc failed for curandState! (code M0005)");
		}
		threadsPerBlock = autoSetBlockSize(random_kernel);
		blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		random_kernel << <blocksPerGrid, threadsPerBlock >> > (state, (unsigned long)time(0));
		generate_random_numbers << <blocksPerGrid, threadsPerBlock >> > (state, data, size);
		cudaFree(state);
		break;
	}
}

cudaMatrix::cudaMatrix(int size, MatrixType type) : rows(size), cols(size) {
	int total_elements = rows * cols;
	cudaError_t err = cudaMalloc((void**)&data, total_elements * sizeof(float));
	if (err != cudaSuccess) {
		throw runtime_error("cudaMalloc failed! (code M0004)");
	}
	int threadsPerBlock = 0;
	int blocksPerGrid = 0;
	switch (type)
	{
	case Zero:
		cudaMemset(data, 0, total_elements * sizeof(float));
		break;
	case Ones:
		threadsPerBlock = autoSetBlockSize(ones_matrix_kernel);
		blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
		ones_matrix_kernel << <blocksPerGrid, threadsPerBlock >> > (data, total_elements);
		break;
	case Identity:
		cudaMemset(data, 0, total_elements * sizeof(float));
		threadsPerBlock = autoSetBlockSize(identity_matrix_kernel);
		blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
		identity_matrix_kernel << <blocksPerGrid, threadsPerBlock >> > (data, rows, cols);
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
		generate_random_numbers << <blocksPerGrid, threadsPerBlock >> > (state, data, total_elements);
		cudaFree(state);
		break;
	}
}

cudaMatrix::cudaMatrix(const cudaMatrix& others) : rows(others.rows), cols(others.cols) {
	cudaError_t err = cudaMalloc((void**)&this->data, static_cast<size_t>(rows) * cols * sizeof(float));
	if (err != cudaSuccess) {
		//throw runtime_error("cudaMalloc failed! (code M0006)");
	}
	cudaMemcpy(this->data, others.data, static_cast<size_t>(rows) * cols * sizeof(float), cudaMemcpyDeviceToDevice);
}

cudaMatrix::~cudaMatrix() {
	cudaMemset(data, 0, static_cast<size_t>(rows) * cols * sizeof(float));
	cudaFree(data);
	rows = 0;
	cols = 0;
}

void cudaMatrix::set(int row, int col, float value) {
	if (row < 0 || row >= rows || col < 0 || col >= cols) {
		throw out_of_range("索引超出范围。");
	}
	cudaMemcpy(data + static_cast<size_t>(row) * cols + col, &value, sizeof(float), cudaMemcpyHostToDevice);
}

void cudaMatrix::setData(const vector<float> v) {
	cudaMemcpy(data, v.data(),
		v.size() * sizeof(float), cudaMemcpyHostToDevice);
}

float cudaMatrix::get(int row, int col) const {
	if (row < 0 || row >= rows || col < 0 || col >= cols) {
		throw out_of_range("索引超出范围。");
	}
	float result = 0.0f;
	cudaMemcpy(&result, data + static_cast<size_t>(row) * cols + col,
		sizeof(float), cudaMemcpyDeviceToHost);
	return result;
}

void cudaMatrix::getData(vector<float>& v, ...) const {
	va_list args;
	va_start(args, &v);
	if (va_arg(args, bool)) { // 如果第一个参数为 true，则强制重新分配内存
		v.resize(static_cast<size_t>(rows) * cols);
	}
	va_end(args);
	cudaMemcpy(v.data(), data,
		v.size() * sizeof(float), cudaMemcpyDeviceToHost);
}

float* cudaMatrix::getDataPtr() const { return data; }

int cudaMatrix::getRows() const { return rows; }

int cudaMatrix::getCols() const { return cols; }

void cudaMatrix::printData() const {
	vector<float> hostData(rows * cols);
	cudaMemcpy(hostData.data(), data,
		static_cast<size_t>(rows) * cols * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			cout << hostData[static_cast<size_t>(i) * cols + j] << " ";
		}
		cout << endl;
	}
	hostData.~vector();
}

float cudaMatrix::norm(int L) const {
	if (rows != 1 && cols != 1) {
		throw invalid_argument("输入不是向量，无法求范数。");
	}
	int size = rows > cols ? rows : cols;
	if (L <= 0) {
		throw invalid_argument("范数阶数必须大于 0。");
	}
	int threadsPerBlock = 768;
	//int threadsPerBlock = autoSetBlockSize(norm_kernel);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	float* vec2 = nullptr;
	cudaMalloc((void**)&vec2, size * sizeof(float));
	norm_kernel << <blocksPerGrid, threadsPerBlock >> > (data, vec2, size, L);
	float* result = nullptr;
	cudaMalloc((void**)&result, sizeof(float));
	cudaMemset(result, 0, sizeof(float));
	reduce_sum << <1, 512, 512 * sizeof(float) >> > (vec2, result, size);
	//cudaDeviceSynchronize();
	float result_host = 0.0f;
	cudaMemcpy(&result_host, result, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(result);
	return powf(result_host, 1.0f / L);
}

cudaMatrix::operator float() const {
	if (rows != 1 || cols != 1) {
		throw invalid_argument("矩阵不是 1x1，无法转换为 float。");
	}
	float result = 0.0f;
	cudaMemcpy(&result, data, sizeof(float), cudaMemcpyDeviceToHost);
	return result;
}

bool cudaMatrix::operator<(const float n) {
	if (rows != 1 || cols != 1) {
		throw invalid_argument("输入不是标量，无法比较。");
	}
	return data[0] < n;
}

bool cudaMatrix::operator<(const cudaMatrix& B) {
	if (rows != 1 || cols != 1 || B.rows != 1 || B.cols != 1) {
		throw invalid_argument("输入不是标量，无法比较。");
	}
	return data[0] < B.data[0];
}

bool cudaMatrix::operator<=(const float n) {
	if (rows != 1 || cols != 1) {
		throw invalid_argument("输入不是标量，无法比较。");
	}
	return data[0] <= n;
}

bool cudaMatrix::operator<=(const cudaMatrix& B) {
	if (rows != 1 || cols != 1 || B.rows != 1 || B.cols != 1) {
		throw invalid_argument("输入不是标量，无法比较。");
	}
	return data[0] <= B.data[0];
}

bool cudaMatrix::operator>(const float n) {
	if (rows != 1 || cols != 1) {
		throw invalid_argument("输入不是标量，无法比较。");
	}
	return data[0] > n;
}

bool cudaMatrix::operator>(const cudaMatrix& B) {
	if (rows != 1 || cols != 1 || B.rows != 1 || B.cols != 1) {
		throw invalid_argument("输入不是标量，无法比较。");
	}
	return data[0] > B.data[0];
}

bool cudaMatrix::operator>=(const float n) {
	if (rows != 1 || cols != 1) {
		throw invalid_argument("输入不是标量，无法比较。");
	}
	return data[0] >= n;
}

bool cudaMatrix::operator>=(const cudaMatrix& B) {
	if (rows != 1 || cols != 1 || B.rows != 1 || B.cols != 1) {
		throw invalid_argument("输入不是标量，无法比较。");
	}
	return data[0] >= B.data[0];
}

cudaMatrix cudaMatrix::add(cudaMatrix& A, cudaMatrix& B) {
	if (A.rows != B.rows || A.cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法相加。");
	}
	cudaMatrix result(A.rows, A.cols);
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

cudaMatrix cudaMatrix::operator=(const cudaMatrix& B) {
	if (this == &B) { return *this; }
	if (rows != B.rows || cols != B.cols) {
		cudaFree(data);
		cudaMalloc((void**)&data, static_cast<size_t>(B.rows) * B.cols * sizeof(float));
		rows = B.rows;
		cols = B.cols;
	}
	cudaMemcpy(data, B.data, static_cast<size_t>(B.rows) * B.cols * sizeof(float), cudaMemcpyDeviceToDevice);
	return *this;
}

cudaMatrix cudaMatrix::operator+(cudaMatrix& B) { return add(*this, B); }

cudaMatrix cudaMatrix::operator+=(const cudaMatrix& B) {
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
	return *this;
}

cudaMatrix cudaMatrix::subtract(const cudaMatrix& A, const cudaMatrix& B) {
	if (A.rows != B.rows || A.cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法相减。");
	}
	cudaMatrix result(A.rows, A.cols);
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

cudaMatrix cudaMatrix::operator-(const cudaMatrix& B) { return subtract(*this, B); }

cudaMatrix cudaMatrix::operator-=(const cudaMatrix& B) {
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
	return *this;
}

cudaMatrix cudaMatrix::multiply(const cudaMatrix& A, const cudaMatrix& B) { // 请不要管这里，这里是屎山
	if (A.rows == 1 && A.cols == 1) {
		cudaMatrix result(B);
		cublasHandle_t handle;
		cublasCreate_v2(&handle);
		cublasSscal_v2(handle, B.rows * B.cols, A.data, result.data, 1);
		cublasDestroy_v2(handle);
		return result;
	}
	if (A.cols != B.rows) {
		throw invalid_argument("矩阵维度不匹配，无法相乘。");
	}
	cudaMatrix temp(A.rows, B.cols);
	cudaMatrix tempA = A.transpose();
	cudaMatrix tempB = B.transpose();
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
	cudaMatrix result = temp.transpose();
	cublasDestroy_v2(handle);
	return result;
}

cudaMatrix operator*(const cudaMatrix& A, const cudaMatrix& B) { return cudaMatrix::multiply(A, B); }

cudaMatrix operator*(float scalar, const cudaMatrix& A) { return A.scalarMultiply(scalar); }

cudaMatrix operator*(const cudaMatrix& A, float scalar) { return A.scalarMultiply(scalar); }

cudaMatrix cudaMatrix::operator*=(const cudaMatrix& B) {
	if (cols != B.rows) {
		throw invalid_argument("矩阵维度不匹配，无法相乘。");
	}
	if (B.cols == 1 && B.rows == 1) {
		cublasHandle_t handle;
		cublasCreate_v2(&handle);
		cublasSscal_v2(handle, rows * cols, B.data, data, 1);
		cublasDestroy_v2(handle);
		return *this;
	}
	if (rows != B.cols) {
		cudaFree(data);
		cols = B.cols;
		cudaMalloc((void**)&data, static_cast<size_t>(rows) * cols * sizeof(float));
	}
	cudaMatrix temp(rows, B.cols);
	cudaMatrix tempA = transpose();
	cudaMatrix tempB = B.transpose();
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
	return *this;
}

cudaMatrix cudaMatrix::operator*=(const float scalar) {
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	cublasSscal_v2(handle, rows * cols, &scalar, data, 1);
	cublasDestroy_v2(handle);
	return *this;
}

cudaMatrix cudaMatrix::operator^(int pows) {
	if (rows != cols) {
		throw invalid_argument("矩阵不是方阵，无法求幂。");
	}
	if (pows < 0) {
		throw invalid_argument("幂次必须大于等于 0。");
	}
	cudaMatrix result(rows, cols, Identity);
	cudaMatrix base = *this;
	while (pows > 0) {
		if (pows % 2 == 1) {
			result = multiply(result, base);
		}
		base = multiply(base, base);
		pows /= 2;
	}
	return result;
}

cudaMatrix cudaMatrix::operator^=(int pows) {
	if (rows != cols) {
		throw invalid_argument("矩阵不是方阵，无法求幂。");
	}
	if (pows < 0) {
		throw invalid_argument("幂次必须大于等于 0。");
	}
	pows--;
	cudaMatrix base = *this;
	while (pows > 0) {
		if (pows % 2 == 1) {
			*this = multiply(*this, base);
		}
		base = multiply(base, base);
		pows /= 2;
	}
	return *this;
}

cudaMatrix cudaMatrix::transpose() const {
	cudaMatrix result(cols, rows);
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

cudaMatrix cudaMatrix::Trans(const cudaMatrix& A) { return A.transpose(); }

cudaMatrix cudaMatrix::operator~() const { return this->transpose(); }

float cudaMatrix::trace() const {
	if (rows != cols) {
		throw invalid_argument("矩阵不是方阵，无法求迹。");
	}
	int size = rows;
	float* trace_array = nullptr;
	cudaMalloc((void**)&trace_array, size * sizeof(float));
	int threadsPerBlock = autoSetBlockSize(get_diag);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	get_diag << <blocksPerGrid, threadsPerBlock >> > (data, trace_array, size);
	cudaDeviceSynchronize();
	float* d_result = nullptr;
	cudaMalloc((void**)&d_result, sizeof(float));
	cudaMemset(d_result, 0, sizeof(float));
	reduce_sum << <1, 512, 512 * sizeof(float) >> > (trace_array, d_result, size);
	float result = 0.0;
	cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(trace_array);
	cudaFree(d_result);
	return (float)result;
}

cudaMatrix cudaMatrix::scalarMultiply(float scalar) const {
	cudaMatrix result(rows, cols);
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	cudaMemcpy(result.data, data, static_cast<size_t>(rows) * cols * sizeof(float), cudaMemcpyDeviceToDevice);
	cublasSscal_v2(handle, rows * cols, &scalar, result.data, 1);
	cublasDestroy_v2(handle);
	return result;
}

cudaMatrix cudaMatrix::matrixDOTmatrix(const cudaMatrix& A, const cudaMatrix& B) {
	if (A.rows != B.rows || A.cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法进行点乘。");
	}
	cudaMatrix result(A.rows, A.cols);
	int size = A.rows * A.cols;
	int threadsPerBlock = autoSetBlockSize(elementwise_multiply_kernel);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	elementwise_multiply_kernel << <blocksPerGrid, threadsPerBlock >> > (A.data, B.data, result.data, size);
	return result;
}

cudaMatrix cudaMatrix::vectorBroadcast2Matrix(const cudaMatrix& sourceVector, const int rows, const int cols) {
	if (sourceVector.cols != 1 && sourceVector.rows != 1) {
		throw invalid_argument("输入矩阵不是向量，无法进行广播。");
	}
	if (rows <= 0 || cols <= 0) {
		throw invalid_argument("广播长度必须大于 0。");
	}
	int size = rows * cols;
	cudaMatrix result(rows, cols);
	if (sourceVector.cols == 1) {
		if (sourceVector.rows != rows)
			throw invalid_argument("向量长度与广播长度不匹配。");
		int threadsPerBlock = autoSetBlockSize(col_vec_broadcast2matrix_kernel);
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		col_vec_broadcast2matrix_kernel << <blocksPerGrid, threadsPerBlock >> > (sourceVector.data, result.data, cols, size);
	}
	else if (sourceVector.rows == 1) {
		if (sourceVector.cols != cols)
			throw invalid_argument("向量长度与广播长度不匹配。");
		int threadsPerBlock = autoSetBlockSize(row_vec_broadcast2matrix_kernel);
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		row_vec_broadcast2matrix_kernel << <blocksPerGrid, threadsPerBlock >> > (sourceVector.data, result.data, cols, size);
	}
	return result;
}

cudaMatrix cudaMatrix::dot(const cudaMatrix& A, const cudaMatrix& B) {
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
		temp1.~cudaMatrix();
		cout << "tempA:" << endl;
		tempA.printData();
	}
	else
		cudaMemcpy(tempA.data, A.data, static_cast<size_t>(A.rows) * A.cols * sizeof(float), cudaMemcpyDeviceToDevice);
	if (B.cols == 1 || B.rows == 1) {
		cudaMatrix temp2 = vectorBroadcast2Matrix(B, row, col);
		cudaMemcpy(tempB.data, temp2.data, static_cast<size_t>(row) * col * sizeof(float), cudaMemcpyDeviceToDevice);
		temp2.~cudaMatrix();
		cout << "tempB:" << endl;
		tempB.printData();
	}
	else
		cudaMemcpy(tempB.data, B.data, static_cast<size_t>(row) * col * sizeof(float), cudaMemcpyDeviceToDevice);
	return matrixDOTmatrix(tempA, tempB);
}

cudaMatrix cudaMatrix::dot(const float scalar, const cudaMatrix& A) { return A.scalarMultiply(scalar); }

cudaMatrix cudaMatrix::dot(const cudaMatrix& A, const float scalar) { return A.scalarMultiply(scalar); }

cudaMatrix cudaMatrix::divide(const cudaMatrix& A, const cudaMatrix& B) {
	if (A.rows != B.rows || A.cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法相除。");
	}
	cudaMatrix result(A.rows, A.cols);
	int size = A.rows * A.cols;
	int threadsPerBlock = autoSetBlockSize(divide_kernel);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	divide_kernel << <blocksPerGrid, threadsPerBlock >> > (A.data, B.data, result.data, size);
	return result;
}

cudaMatrix cudaMatrix::operator/(const cudaMatrix& B) { return divide(*this, B); }

cudaMatrix cudaMatrix::operator/(const float scalar) {
	float invScalar = 1.0f / scalar;
	cudaMatrix result(rows, cols);
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	cudaMemcpy(result.data, data, static_cast<size_t>(rows) * cols * sizeof(float), cudaMemcpyDeviceToDevice);
	cublasSscal_v2(handle, rows * cols, &invScalar, result.data, 1);
	cublasDestroy_v2(handle);
	return result;
}

cudaMatrix cudaMatrix::operator/=(const cudaMatrix& B) {
	if (rows != B.rows || cols != B.cols) {
		throw invalid_argument("矩阵维度不匹配，无法相除。");
	}
	int size = rows * cols;
	int threadsPerBlock = autoSetBlockSize(divide_kernel);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	divide_kernel << <blocksPerGrid, threadsPerBlock >> > (data, B.data, data, size);
	return *this;
}

cudaMatrix cudaMatrix::operator/=(const float scalar) {
	float invScalar = 1.0f / scalar;
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	cublasSscal_v2(handle, rows * cols, &invScalar, data, 1);
	cublasDestroy_v2(handle);
	return *this;
}

cudaMatrix cudaMatrix::solveSparseSLE(cudaMatrix& A, cudaMatrix& b) {
	if (A.rows != b.rows) {
		throw invalid_argument("矩阵维度不匹配，无法求解稀疏线性方程组。");
	}
	if (b.cols != 1) {
		throw invalid_argument("右侧矩阵不是列向量，无法求解稀疏线性方程组。");
	}
	cudaMatrix x(A.rows, 1, Ones);
	cudaMatrix r = b - A * x;
	cudaMatrix p = r;
	cudaMatrix r_old = r;
	float r_norm2 = ~r * r;
	for (int i = 0; i < 1e6; i++)
	{
		cudaMatrix Ap = A * p;
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

cudaMatrix cudaMatrix::operator| (cudaMatrix& b) { return solveSparseSLE(*this, b); }

float cudaMatrix::det() const {
	if (rows != cols) {
		throw invalid_argument("矩阵不是方阵，无法求行列式。");
	}
	int size = rows;
	cusolverDnHandle_t handle;
	cusolverDnCreate(&handle);

	// 创建一个副本以避免修改原始矩阵
	cudaMatrix temp(*this);

	// 分配内存以存储Pivots和Info
	int* Pivots = nullptr;
	int* Info = nullptr;
	cudaMalloc((void**)&Pivots, size * sizeof(int));
	cudaMalloc((void**)&Info, sizeof(int));

	// LU分解
	int workspace_size = 0;
	cusolverDnSgetrf_bufferSize(handle, size, size, temp.data, size, &workspace_size);
	float* workspace = nullptr;
	cudaMalloc((void**)&workspace, workspace_size * sizeof(float));
	cusolverDnSgetrf(handle, size, size, temp.data, size, workspace, Pivots, Info);

	// 计算行列式
	float det = 1.0f;
	float* diag = nullptr;
	cudaMalloc((void**)&diag, size * sizeof(float));
	int threadsPerBlock = autoSetBlockSize(get_diag);
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	get_diag << <blocksPerGrid, threadsPerBlock >> > (temp.data, diag, size);
	cudaDeviceSynchronize();
	vector<float> hostDiag(size);
	cudaMemcpy(hostDiag.data(), diag, size * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; ++i) { det *= hostDiag[i]; }

	// 检查Pivots以确定行列式的符号
	int* h_Pivots = new int[size];
	cudaMemcpy(h_Pivots, Pivots, size * sizeof(int), cudaMemcpyDeviceToHost);
	int pivotSign = 1;
	for (int i = 0; i < size; ++i) {
		if (h_Pivots[i] != (i + 1)) {
			pivotSign *= -1;
		}
	}
	det *= pivotSign;

	// 释放内存
	delete[] h_Pivots;
	cudaFree(Pivots);
	cudaFree(Info);
	cudaFree(workspace);
	cusolverDnDestroy(handle);

	return det;
}

float cudaMatrix::det(const cudaMatrix& A) { return A.det(); }

cudaMatrix cudaMatrix::diag(vector<int> offset, ...) {
	int num = offset.size();
	va_list args;
	va_start(args, offset);
	vector<cudaMatrix> arg(num);
	arg[0] = va_arg(args, cudaMatrix);
	for (int i = 0; i < num; ++i) {
		arg[i] = cudaMatrix(va_arg(args, cudaMatrix));
		//if (arg[i].data == nullptr)
			//throw invalid_argument("输入矩阵指针为空。");
	}
	va_end(args);
	int size = arg[0].rows > arg[0].cols ? arg[0].rows : arg[0].cols;
	cudaMatrix result(size);
	for (int i = 0; i < num; ++i) {
		int threadsPerBlock = autoSetBlockSize(fill_diag_kernel);
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
		fill_diag_kernel << <blocksPerGrid, threadsPerBlock >> > (result.data, arg[i].data, size, offset[i]);
	}
	return result;
}

cudaMatrix::ElementProxy cudaMatrix::operator()(int row, int col) {
	if (row < 0 || row >= rows || col < 0 || col >= cols) {
		throw std::out_of_range("索引超出范围。");
	}
	return ElementProxy(*this, row, col);
}

cudaMatrix::ElementProxy::~ElementProxy() {
	row = 0;
	col = 0;
	mat = cudaMatrix();
}

cudaMatrix::ElementProxy::operator float() const { return mat.get(row, col); }

cudaMatrix::ElementProxy& cudaMatrix::ElementProxy::operator=(float value) {
	mat.set(row, col, value);
	return *this;
}
