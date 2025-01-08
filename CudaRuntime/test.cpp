#include <iostream>
#include <vector>
#include "cuda_matrix.h"

using namespace std;

int main() {
	// 初始化CUDA设备
	cudaSetDevice(0);

	// 定义两个矩阵A和B
	vector<float> hostA = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
	vector<float> hostB = { 3.0f, 4.0f, 6.0f };

	// 将数据从主机复制到设备
	cudaMatrix A(1, hostA.size());
	cudaMatrix B(1, hostB.size());
	A.setData(hostA);
	B.setData(hostB);

	// 调用setdiff函数
	cudaMatrix result = cudaMatrix::setdiff(A, B);

	// 将结果从设备复制到主机
	vector<float> hostResult(result.getCols());
	result.getData(hostResult);

	// 打印结果
	cout << "Result of setdiff(A, B): ";
	for (float val : hostResult) {
		cout << val << " ";
	}
	cout << endl;
	system("pause");
	return 0;
}