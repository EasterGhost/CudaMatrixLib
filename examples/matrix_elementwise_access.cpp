/**
 * @file matrix_elementwise_access.cpp
 * @author Andrew Elizabeth (2934664277@qq.com)
 * @brief Example of accessing elements of a matrix using the CudaMatrix class
 * @version 1.0
 * @date 2025-02-13
 */
#include "cuda_matrix.cuh"
#include "cuda_matrix.cu"

int main()
{
    CudaMatrix<float> A(3, 4, Random); /// Generate a random matrix A
    /// Modify some elements of matrix A using the () operator
    A(0, 0) = 1.4f;
    A(1, 2) = 2.3f;
    A(2, 3) = 3.8f;
    cout << "Random Matrix A:" << endl;
    A.print();
    cout << "A(0, 0): " << A(0, 0) << endl; /// Access elements of matrix A using the () operator
    return 0;
}