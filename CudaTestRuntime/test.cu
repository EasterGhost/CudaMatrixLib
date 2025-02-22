/**
 * @file matrix_type_conversion.cpp
 * @author Andrew Elizabeth (2934664277@qq.com)
 * @brief Example of converting a matrix of one type to another using the cumatrix class
 * @version 1.0
 * @date 2025-02-13
 */
#include "cuda_matrix.cuh"
#include "cuda_matrix.cu"

int main()
{
	constexpr uint32_t n = 1000;
	constexpr int max_loop = 1000;
	cumatrix<float> mat1(n, Random);
	cumatrix<float> mat2(n, Random);
	cumatrix<float> mat3(n); // result of cublas
	cout << "cublas multiply test:" << endl;
	clock_t start = clock();
	for (int loop = 0; loop < max_loop; loop++)
	{
		cublasHandle_t handle;
		cublasCreate_v2(&handle);
		const float alpha = 1.0f;
		const float beta = 0.0f;
		cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, mat1.data(), n, mat2.data(), n, &beta, mat3.data(), n);
		//cout << "Matrix res of cublas:" << endl;
		//mat3.print();
		cublasDestroy_v2(handle);
		if (loop % 100 == 0)
			cout << "loop: " << loop << endl;
	}
	clock_t time_used_cublas = clock() - start;
	cumatrix<float> mat4(n); // result of kernel_float
	dim3 grid(n, n);
	int block_size = autoSetBlockSize(matrix_multiply_kernel3<float, float, float>);
	size_t shared_memory_size = block_size * sizeof(float);
	cout << "kernel multiply test(double + fma):" << endl;
	start = clock();
	for (int loop = 0; loop < max_loop * 10; loop++)
	{
		matrix_multiply_kernel3<float, float, float>
			<< <grid, block_size, shared_memory_size >> > (mat1.data(), mat2.data(), mat4.data(), n, n, n);
		cudaDeviceSynchronize();
		if (loop % 100 == 0)
			cout << "loop: " << loop << endl;
	}
	clock_t time_used_fma = clock() - start;
	//cout << "Matrix res of kernel:" << endl;
	//mat4.print();
	cumatrix<float> mat5(n); // result of kernel_double
	cout << "kernel multiply test(double):" << endl;
	start = clock();
	for (int loop = 0; loop < max_loop * 10; loop++)
	{
	matrix_multiply_kernel2<float, float, float>
		<< <grid, block_size, shared_memory_size >> > (mat1.data(), mat2.data(), mat5.data(), n, n, n);
	cudaDeviceSynchronize();
	if (loop % 100 == 0)
		cout << "loop: " << loop << endl;
	}
	clock_t time_used_without_fma = clock() - start;
	//mat4.print();
	vector<float> vec1(n * n);
	vector<float> vec2(n * n);
	cumatrix<float> mat6(n);
	int threadsPerBlock = autoSetBlockSize(elementwise_subtract_kernel<float, float, float>);
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
		elementwise_subtract_kernel<float, float, float>
		<< <blocksPerGrid, threadsPerBlock >> > (mat4.data(), mat3.data(), mat6.data(), n * n);
	cumatrix<float> mat7(n);
	start = clock();
	for (int loop = 0; loop < 1000; loop++)
		elementwise_subtract_kernel<float, float, float>
		<< <blocksPerGrid, threadsPerBlock >> > (mat5.data(), mat3.data(), mat7.data(), n * n);
	mat6.get_data(vec1);
	mat7.get_data(vec2);
	float err1 = 0;
	float err2 = 0;
	for (int i = 0; i < n * n; i++)
	{
		err1 += abs(vec1[i]);
		err2 += abs(vec2[i]);
	}
	cout << "Time used by cublas: " << time_used_cublas << "ms" << endl;
	cout << "Error of kernel_fma: " << err1 << endl;
	cout << "Time used by kernel with fma: " << time_used_fma << "ms" << endl;
	cout << "Error of kernel_default: " << err2 << endl;
	cout << "Time used by kernel without fma: " << time_used_without_fma << "ms" << endl;
	cout << "Speedup of kernel with fma: " << (float)time_used_without_fma / (float)time_used_fma<< endl;
	system("pause");
	return 0;
}