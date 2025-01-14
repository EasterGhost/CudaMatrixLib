#include "cuda_matrix.h"

__global__ static void fillKKernel(int nelx, int nely, float* x, float* KE, float* K) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < nelx) {
		if (idy < nely) {
			int e = idx * nely + idy;
			int n1 = (nely + 1) * idx + idy;
			int n2 = (nely + 1) * (idx + 1) + idy;
			int edof[] = { 2 * n1 - 1, 2 * n1, 2 * n2 - 1, 2 * n2, 2 * n2 + 1, 2 * n1 + 1, 2 * n1 + 2 };
			for (int i = 0; i < 8; i++) {
				for (int j = 0; j < 8; j++) {
					atomicAdd(&K[edof[i] * 2 * (nelx + 1) + edof[j]], x[e] * KE[i * 8 + j]);
				}
			}
		}
	}
}

static cudaMatrix lk() {
	cudaMatrix KE(8);
	float E = 1.0f;
	float nu = 0.3f;
	vector<float> k(8);
	k[0] = 0.5f - nu / 6.0f;
	k[1] = 1.0f / 8.0f + nu / 8.0f;
	k[2] = -1.0f / 4.0f - nu / 12.0f;
	k[3] = -1.0f / 8.0f + 3.0f * nu / 8.0f;
	k[4] = -1.0f / 4.0f + nu / 12.0f;
	k[5] = -1.0f / 8.0f - nu / 8.0f;
	k[6] = nu / 6.0f;
	k[7] = 1.0f / 8.0f - 3.0f * nu / 8.0f;

	// 将系数数组 k 拷贝到设备端
	float* d_k;
	cudaMalloc(&d_k, 8 * sizeof(float));
	cudaMemcpy(d_k, k.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);

	// 获取 KE 矩阵的设备数据指针
	float* KE_data = KE.getDataPtr();

	// 定义 CUDA 内核函数，填充 KE 矩阵
	auto fillKEKernel = [E, nu] __global__(float* KE_data, const float* k) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx < 64) { // 8x8 矩阵，共 64 个元素
			int i = idx / 8;
			int j = idx % 8;

			// 定义 KE 矩阵的填充值（按照 MATLAB 代码的填充方式）
			__shared__ float KE_values[8][8]{};
			if (threadIdx.x < 8 && threadIdx.y < 8) {
				KE_values[threadIdx.x][threadIdx.y] = 0.0f;
			}
			__syncthreads();

			float E_factor = E / (1.0f - nu * nu);
			float k_val[] = {
				k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
				k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2],
				k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1],
				k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4],
				k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3],
				k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6],
				k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5],
				k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]
			};

			KE_data[idx] = E_factor * k_val[idx];
		}
	};

	// 启动 CUDA 内核
	int threadsPerBlock = 64;
	int blocksPerGrid = (64 + threadsPerBlock - 1) / threadsPerBlock;
	fillKEKernel << <blocksPerGrid, threadsPerBlock >> > (KE_data, d_k);
	cudaFree(d_k);
	return KE;
}

cudaMatrix FE(int nelx, int nely, cudaMatrix& x, float penal) {
	cudaMatrix KE = lk();
	cudaMatrix K(2 * (nelx + 1) * (nely + 1));
	cudaMatrix F(2 * (nelx + 1) * (nely + 1), 1);
	cudaMatrix U(2 * (nelx + 1) * (nely + 1), 1, Ones);
	fillKKernel << <dim3((nelx + 1) / 16 + 1, (nely + 1) / 16 + 1), dim3(16, 16) >> > (nelx, nely, x.getDataPtr(), KE.getDataPtr(), K.getDataPtr());
	F(2 * (nely + 1) * nelx + nely, 0) = -1.0f;
	cudaMatrix fixedDOFs(2 * (nely + 1), 1);
	cudaMatrix allDOFs(2 * (nely + 1) * (nelx + 1), 1);
	auto fillFixedDOFsKernel = [nely] __global__(float* fixedDOFs_data) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx < 2 * (nely + 1)) {
			fixedDOFs_data[idx] = idx;
		}
	};
	auto fillAllDOFsKernel = [nely, nelx] __global__(float* allDOFs_data) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx < 2 * (nely + 1) * (nelx + 1)) {
			allDOFs_data[idx] = idx;
		}
	};
	cudaMatrix freeDOFs = cudaMatrix::setdiff(allDOFs, fixedDOFs);

}