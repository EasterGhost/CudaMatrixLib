#include "TemplateCudaMatrix.cuh"
#include "TemplateCudaMatrix.cu"

int main()
{
	clock_t start = clock();
	CudaMatrix<int> m(5000, Random);
	clock_t end = clock();
	cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
	//m.print();
	system("pause");
	return 0;
}