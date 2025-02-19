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
	constexpr int num_tests = 1e4;
	constexpr int size = 10000;
	int block_size = autoSetBlockSize(elementwise_add_kernel<float, double, float>);
	int convert_block_size = autoSetBlockSize(convert_kernel<float, double>);
	int grid_size = (size + block_size - 1) / block_size;
	int convert_grid_size = (size + convert_block_size - 1) / convert_block_size;
	clock_t start, end;
	cout << "----------Basic Matrix Function Test----------" << endl
		<< "Matrix Size: " << size << "x" << size << endl << "Loop Times: " << num_tests << endl
		<< "----------Matrix Type Conversion Test----------" << endl << "0% Done";
	start = clock();
	for (int i = 1; i <= num_tests; i++)
	{
		cumatrix<float> mat1(size, Random);
		cumatrix<double> mat2(size);
		convert_kernel<float, double> << <convert_grid_size, convert_block_size >> >
			(mat1.data(), mat2.data(), size * size);
		cudaDeviceSynchronize();
		if (i % (num_tests / 100) == 0)
			cout << endl << i / (num_tests / 100) << "% Done";
		if (i % (num_tests / 1000) == 0)
			cout << ".";
	}
	end = clock();
	cout << endl << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl
		<< "Loop Times: " << num_tests << endl
		<< "----------Matrix Type Conversion Test Finished----------" << endl << endl
		<< "----------Matrix Element-wise Addition Test----------" << endl << "0% Done";
	start = clock();
	for (int i = 1; i <= num_tests; i++)
	{
		cumatrix<float> mat1(size, Random);
		cumatrix<double> mat2(size, Random);
		cumatrix<float> mat3(size);
		elementwise_add_kernel<float, double, float> << <grid_size, block_size >> >
			(mat1.data(), mat2.data(), mat3.data(), size * size);
		cudaDeviceSynchronize();
		if (i % (num_tests / 100) == 0)
			cout << endl << i / (num_tests / 100) << "% Done";
		if (i % (num_tests / 10) == 0)
			cout << ".";
	}
	end = clock();
	cout << endl << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl
		<< "Loop Times: " << num_tests << endl
		<< "----------Matrix Element-wise Addition Test Finished----------" << endl << endl
		<< "----------Matrix Element-wise Subtraction Test----------" << endl << "0% Done";
	start = clock();
	for (int i = 1; i <= num_tests; i++)
	{
		cumatrix<float> mat1(size, Random);
		cumatrix<double> mat2(size, Random);
		cumatrix<float> mat3(size);
		elementwise_subtract_kernel<float, double, float> << <grid_size, block_size >> >
			(mat1.data(), mat2.data(), mat3.data(), size * size);
		cudaDeviceSynchronize();
		if (i % (num_tests / 100) == 0)
			cout << endl << i / (num_tests / 100) << "% Done";
		if (i % (num_tests / 10) == 0)
			cout << ".";
	}
	end = clock();
	cout << endl << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl
		<< "Loop Times: " << num_tests << endl
		<< "----------Matrix Element-wise Subtraction Test Finished----------" << endl << endl
		<< "----------Matrix Basic Function Test Finished----------" << endl;
	system("pause");
	return 0;
}