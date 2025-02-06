#include "cuda_matrix.cu"
#include "cuda_matrix.cuh"

int main()
{
	//randomMatrixGenerationTest();
	//randomMatrixGenerationTestDemo();
	//qrandomMatrixGenerationTest();
	//qrandomMatrixGenerationTestDemo();
	CudaMatrix<float> A(5, Random);
	A(2, 3) = 1.2f;
	A(1, 4) = 2.6f;
	A(0, 0) = 3.0f;
	A.print();
	CudaMatrix<int> B = static_cast<CudaMatrix<int>>(A);
	
	B.print();
	system("pause");
	return 0;
}