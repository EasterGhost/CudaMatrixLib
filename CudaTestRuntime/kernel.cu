#include "TemplateCudaMatrix.cuh"

int main()
{
	CudaMatrix<int> m(3, 3, Identity);
	return 0;
}