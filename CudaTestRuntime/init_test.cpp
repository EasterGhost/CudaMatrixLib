#include "cuda_matrix.cuh"
#include <thread>

static void init_test()
{
	for (int i = 0; i < 5; i++) {
		clock_t start = clock();
		cublasHandle_t handle;
		cusolverDnHandle_t solver_handle;
		cublasCreate_v2(&handle);
		cusolverDnCreate(&solver_handle);
		cublasDestroy_v2(handle);
		cusolverDnDestroy(solver_handle);

		clock_t end = clock();
		cout << "Time used in init test: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
		start = clock();
		thread t1([]() {
			cublasHandle_t handle1;
			cublasCreate_v2(&handle1);
			cublasDestroy_v2(handle1);
			});
		thread t2([]() {
			cusolverDnHandle_t solver_handle1;
			cusolverDnCreate(&solver_handle1);
			cusolverDnDestroy(solver_handle1);
			});
		t1.join();
		t2.join();
		end = clock();
		cout << "Time used in init test (thread): " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
	}return;
}