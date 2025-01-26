#include "TemplateCudaMatrix.cuh"
#include "TemplateCudaMatrix.cu"

int main()
{
	cout << "--------Random matrix generation test.--------" << endl;
	cout << "Press any key to start." << endl;
	system("pause");
	cout << endl << "----------------------------------------" << endl;
	int n = 5000;
	int loop = 1000;
	clock_t start, end;
	cout << "Int Matrix Test Start." << endl;
	start = clock();
	cout << "0% Done";
	for (int i = 1; i <= loop; i++)
	{
		CudaMatrix<int> m(n, Random);
		if (i % 100 == 0)
			cout << endl << i / 10 << "% Done";
		if (i % 10 == 0)
			cout << ".";
	}
	end = clock();
	cout << endl << "Time used in int matrix generation: "
		<< (double)(end - start) / 1000000 << "s" << endl
		<< "----------------------------------------" << endl
		<< "Float Matrix Test Start." << endl;
	start = clock();
	cout << "0% Done";
	for (int i = 1; i <= loop; i++)
	{
		CudaMatrix<float> m(n, Random);
		if (i % 100 == 0)
			cout << endl << i / 10 << "% Done";
		if (i % 10 == 0)
			cout << ".";
	}
	end = clock();
	cout
		<< endl << "Time used in float matrix generation: "
		<< (double)(end - start) / 1000000 << "s" << endl
		<< "----------------------------------------" << endl
		<< "Double Matrix Test Start." << endl;
	start = clock();
	cout << "0% Done";
	for (int i = 1; i <= loop; i++)
	{
		CudaMatrix<double> m(n, Random);
		if (i % 100 == 0)
			cout << endl << i / 10 << "% Done";
		if (i % 10 == 0)
			cout << ".";
	}
	end = clock();
	cout << endl << "Time used in double matrix generation: "
		<< (double)(end - start) / 1000000 << "s" << endl;
	cout << "--------Random matrix generation test end.--------" << endl;
	system("pause");
	return 0;
}