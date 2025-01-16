#include "TemplateCudaMatrix.cuh"
#include "TemplateCudaMatrix.cu"

int main()
{
	CudaMatrix<float> m(3, 3, Random);
	m.print();
	system("pause");
	return 0;
}