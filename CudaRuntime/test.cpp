#include "cuda_matrix.h"

static float kahan_add(vector<float> arr, int size) {
	float sum = 0.0f;
	float c = 0.0f;
	for (int i = 0; i < size; i++) {
		float y = arr[i] * arr[i] - c;
		float t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}
	return sum;
}

int main() {
	vector<int> offsets = { 0, 1, -1 };
	vector<float> diag1 = { 1.0f, 2.0f, 3.0f };
	vector<float> diag2 = { 4.0f, 5.0f };
	vector<float> diag3 = { 0.0f, 6.0f, 7.0f };

	cudaMatrix d1(diag1.size(), 1);
	d1.setData(diag1);
	cudaMatrix d2(diag2.size(), 1);
	d2.setData(diag2);
	cudaMatrix d3(diag3.size(), 1);
	d3.setData(diag3);

	cudaMatrix result = cudaMatrix::diag(offsets, diag1, diag2, diag3);
	result.printData();
	cout << "det = " << result.det() << endl;
	result.resize(4, 5);
	result.printData();
	result.resize(2, 2);
	result.printData();
	//cout << "norm = " << result.norm(2) << endl;
	system("pause");
	return 0;
}