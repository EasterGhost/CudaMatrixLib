#include <iostream>
#include <vector>
#include "cuda_matrix.h"

using namespace std;

int main() {
	// ��ʼ��CUDA�豸
	cudaSetDevice(0);

	// ������������A��B
	vector<float> hostA = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
	vector<float> hostB = { 3.0f, 4.0f, 6.0f };

	// �����ݴ��������Ƶ��豸
	cudaMatrix A(1, hostA.size());
	cudaMatrix B(1, hostB.size());
	A.setData(hostA);
	B.setData(hostB);

	// ����setdiff����
	cudaMatrix result = cudaMatrix::setdiff(A, B);

	// ��������豸���Ƶ�����
	vector<float> hostResult(result.getCols());
	result.getData(hostResult);

	// ��ӡ���
	cout << "Result of setdiff(A, B): ";
	for (float val : hostResult) {
		cout << val << " ";
	}
	cout << endl;
	system("pause");
	return 0;
}