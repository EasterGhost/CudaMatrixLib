/**
 * @file cuda_matrix_generation.cpp
 * @author Andrew Elizabeth (2934664277@qq.com)
 * @brief Example of generating a random, identity, and zero matrix using the CudaMatrix class
 * @version 1.0
 * @date 2025-02-13
 * @note 
 */
#include "cuda_matrix.cuh"
#include "cuda_matrix.cu"

void cuda_matrix_generation()
{
	CudaMatrix<int> A(3, 4, Random); // Generate a random matrix A with 3 rows and 4 columns
	CudaMatrix<float> B(3, Identity); // Generate an identity matrix B with 3 rows and 3 columns
	CudaMatrix<double> C(3); // Generate a zero matrix C with 3 rows and 3 columns
	vector<int> v = {1, 2, 3};
	CudaMatrix<int> D(3, 1, v); // Generate a column vector matrix D with 3 rows, initialized with vector v
	CudaMatrix<int> E(1, 4, v); // Generate a row vector matrix E with 4 columns, initialized with vector v
	CudaMatrix<int> F(A); // Generate a copy matrix F of matrix A
	cout << "Random Matrix A:" << endl;
	A.print();
	cout << "Identity Matrix B:" << endl;
	B.print();
	cout << "Zero Matrix C:" << endl;
	C.print();
	cout << "ColVector Matrix D:" << endl;
	D.print();
	cout << "RowVector Matrix E:" << endl;
	E.print();
	cout << "Copy Matrix F:" << endl;
	F.print();
	return;
}