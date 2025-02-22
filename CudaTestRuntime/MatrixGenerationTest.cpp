﻿#include "cuda_matrix.cuh"
#include "vector_operator.cpp"

static void updateStandardTimeList
(vector<clock_t>& standard_time_list, const clock_t time_used_total, const size_t debuglength)
{
	if (standard_time_list.size() != debuglength)
		throw runtime_error("Invalid standard time list.");
	standard_time_list[0] = time_used_total;
	standard_time_list[1] = time_used_init;
	standard_time_list[2] = time_used_gen_init;
	standard_time_list[3] = time_used_switch_type;
	standard_time_list[4] = time_used_setblock;
	standard_time_list[5] = time_used_gen;
	standard_time_list[6] = time_used_end;
	return;
}

static void printMatrixGenerationDebugInfo
(const int loop, const vector<clock_t>& standard_time_list, const string Type, const size_t debuglength)
{
	if (standard_time_list.size() != debuglength)
		throw runtime_error("Invalid standard time list.");
	cout << endl << "Time used in " << Type << " matrix generation : "
		<< (double)standard_time_list[0] / (CLOCKS_PER_SEC * loop) << "s" << endl
		<< "time used in matrix generation (init): "
		<< (double)standard_time_list[1] / (CLOCKS_PER_SEC * loop) << "s" << endl
		<< "time used in matrix generation (gen init): "
		<< (double)standard_time_list[2] / (CLOCKS_PER_SEC * loop) << "s" << endl
		<< "time used in matrix generation (switch type): "
		<< (double)standard_time_list[3] / (CLOCKS_PER_SEC * loop) << "s" << endl
		<< "time used in matrix generation (setblock): "
		<< (double)standard_time_list[4] / (CLOCKS_PER_SEC * loop) << "s" << endl
		<< "time used in matrix generation (gen): "
		<< (double)standard_time_list[5] / (CLOCKS_PER_SEC * loop) << "s" << endl
		<< "time used in matrix generation (end): "
		<< (double)standard_time_list[6] / (CLOCKS_PER_SEC * loop) << "s" << endl
		<< "----------------------------------------" << endl;
	return;
}

static void randomMatrixGenerationTestDemo()
{
	CudaMatrix<int> m(10, Random);
	cout << "Int Matrix Test." << endl;
	m.print();
	CudaMatrix<float> m1(10, Random);
	cout << "Float Matrix Test." << endl;
	m1.print();
	CudaMatrix<double> m2(10, Random);
	cout << "Double Matrix Test." << endl;
	m2.print();
	CudaMatrix<uint8_t> m3(10, Random);
	cout << "Char Matrix Test." << endl;
	m3.print();
	return;
}

static void randomMatrixGenerationTest()
{
	constexpr int n = 10000;
	constexpr int loop = 1000;
	constexpr int debuglength = 7;
	cout << "--------Random matrix generation test.--------" << endl;
	cout << "Basic information:" << endl
		<< "Matrix size: " << n << "x" << n << endl
		<< "Loop: " << loop << endl;
	system("pause");
	cout << endl << "----------------------------------------" << endl;
	vector<vector<clock_t>> standard_time_list(3);
	for (int i = 0; i < 3; i++)
		standard_time_list[i].resize(debuglength);
	clock_t start;
	cout << "Int Matrix Test Start." << endl << "0% Done";
	start = clock();
	for (int i = 1; i <= loop; i++)
	{
		CudaMatrix<int> m(n, QuasiRandom);
		if (i % (loop / 10) == 0)
			cout << endl << i / (loop / 100) << "% Done";
		if (i % (loop / 100) == 0)
			cout << ".";
	}
	updateStandardTimeList(standard_time_list[0], clock() - start, debuglength);
	printMatrixGenerationDebugInfo(loop, standard_time_list[0], "int", debuglength);
	_sleep(10000);
	cout << "Float Matrix Test Start." << endl << "0% Done";
	start = clock();
	for (int i = 1; i <= loop; i++)
	{
		CudaMatrix<float> m(n, Random);
		if (i % (loop / 10) == 0)
			cout << endl << i / (loop / 100) << "% Done";
		if (i % (loop / 100) == 0)
			cout << ".";
	}
	updateStandardTimeList(standard_time_list[1], clock() - start, debuglength);
	standard_time_list[1][0] += standard_time_list[0][0];
	printMatrixGenerationDebugInfo
	(loop, standard_time_list[1] - standard_time_list[0], "float", debuglength);
	standard_time_list[1][0] -= standard_time_list[0][0];
	_sleep(10000);
	cout << "Double Matrix Test Start." << endl << "0% Done";
	start = clock();
	for (int i = 1; i <= loop; i++)
	{
		CudaMatrix<double> m(n, Random);
		if (i % (loop / 10) == 0)
			cout << endl << i / (loop / 100) << "% Done";
		if (i % (loop / 100) == 0)
			cout << ".";
	}
	updateStandardTimeList(standard_time_list[2], clock() - start, debuglength);
	standard_time_list[2][0] += standard_time_list[1][0];
	printMatrixGenerationDebugInfo
	(loop, standard_time_list[2] - standard_time_list[1], "double", debuglength);
	standard_time_list[2][0] -= standard_time_list[1][0];
	_sleep(10000);
	cout << "--------Random matrix generation test end.--------" << endl;
	return;
}

static void qrandomMatrixGenerationTest()
{
	constexpr int n = 10000;
	constexpr int loop = 1000;
	constexpr int debuglength = 7;
	cout << "--------QuasiRandom matrix generation test.--------" << endl;
	cout << "Basic information:" << endl
		<< "Matrix size: " << n << "x" << n << endl
		<< "Loop: " << loop << endl;
	system("pause");
	cout << endl << "----------------------------------------" << endl;
	vector<vector<clock_t>> standard_time_list(3);
	for (int i = 0; i < 3; i++)
		standard_time_list[i].resize(debuglength);
	clock_t start;
	cout << "Float Matrix Test Start." << endl << "0% Done";
	start = clock();
	for (int i = 1; i <= loop; i++)
	{
		CudaMatrix<float> m(n, QuasiRandom);
		if (i % (loop / 10) == 0)
			cout << endl << i / (loop / 100) << "% Done";
		if (i % (loop / 100) == 0)
			cout << ".";
	}
	updateStandardTimeList(standard_time_list[0], clock() - start, debuglength);
	printMatrixGenerationDebugInfo(loop, standard_time_list[0], "float", debuglength);
	_sleep(10000);
	cout << "Double Matrix Test Start." << endl << "0% Done";
	start = clock();
	for (int i = 1; i <= loop; i++)
	{
		CudaMatrix<double> m(n, QuasiRandom);
		if (i % (loop / 10) == 0)
			cout << endl << i / (loop / 100) << "% Done";
		if (i % (loop / 100) == 0)
			cout << ".";
	}
	updateStandardTimeList(standard_time_list[1], clock() - start, debuglength);
	printMatrixGenerationDebugInfo(loop, standard_time_list[1], "double", debuglength);
	return;
}

static void qrandomMatrixGenerationTestDemo()
{
	CudaMatrix<int> m(10, QuasiRandom);
	cout << "Int Matrix Test." << endl;
	m.print();
	CudaMatrix<float> m1(10, QuasiRandom);
	cout << "Float Matrix Test." << endl;
	m1.print();
	CudaMatrix<double> m2(10, QuasiRandom);
	cout << "Double Matrix Test." << endl;
	m2.print();
	return;
}