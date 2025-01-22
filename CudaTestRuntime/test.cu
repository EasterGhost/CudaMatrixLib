#include "TemplateCudaMatrix.cuh"
#include "TemplateCudaMatrix.cu"

int main()
{
	CudaMatrix<int> m(3, Random);
	m.print();
	system("pause");
	return 0;
}