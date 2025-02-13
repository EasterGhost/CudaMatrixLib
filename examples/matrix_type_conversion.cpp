/**
 * @file matrix_type_conversion.cpp
 * @author Andrew Elizabeth (2934664277@qq.com)
 * @brief Example of converting a matrix of one type to another using the CudaMatrix class
 * @version 1.0
 * @date 2025-02-13
 */
#include "cuda_matrix.cuh"
#include "cuda_matrix.cu"

int main()
{
    CudaMatrix<double> A(3, 4, Random); /// Generate a random matrix A
    /// Modify some elements of matrix A
    A(0, 0) = 1.4;
    A(1, 2) = 2.3;
    A(2, 3) = 3.8;
    cout << "Random Matrix A:" << endl;
    A.print();
    cout << "A(0, 0): " << A(0, 0) << endl;
    CudaMatrix<int> B = static_cast<CudaMatrix<int>>(A); /// Convert matrix A to an integer matrix B using the static_cast operator
    cout << "Matrix B:" << endl;
    B.print();
    CudaMatrix<int> C(3, 4, Zero);
    C = A; /// Convert matrix A to an integer matrix C using the assignment operator
    cout << "Matrix C:" << endl;
    C.print();
    return 0;
}