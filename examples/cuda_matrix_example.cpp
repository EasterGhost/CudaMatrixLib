/**
* @file cuda_matrix_example.cpp
* @brief CUDA ������ʾ��
* @note ��֧�� float ����
* @note ʹ�� cuBLAS ʵ�ֲ��־�������
* @note ��ʾ��չʾ�� CUDA ������Ļ����÷�
*/
#include "cuda_matrix.h"

int main() {
	cudaMatrix A(3, 3, MatrixType::Random);
	cudaMatrix B(3, MatrixType::Ones);
	cudaMatrix C = A + B;
	cout << "A = " << endl;
	A.printData();
	cout << "B = " << endl;
	B.printData();
	cout << "C = A + B = " << endl;
	C.printData();
	cudaMatrix D = A * C;
	cout << "D = A * C = " << endl;
	D.printData();
	cudaMatrix E = A ^ 2;
	cout << "E = A^2 = " << endl;
	E.printData();
	cudaMatrix F = ~A;
	cout << "F = A' = " << endl;
	F.printData();
	cudaMatrix G(3, MatrixType::Identity);
	cudaMatrix H = dot(A, G);
	cout << "H = A.*G = " << endl;
	H.printData();
	cudaMatrix I = A / B;
	cout << "I = A./B = " << endl;
	I.printData();
	cudaMatrix b(3, 1, MatrixType::Random);
	cout << "b = " << endl;
	b.printData();
	cudaMatrix x = A | b;
	cout << "x = " << endl;
	x.printData();
	system("pause");
	return 0;
}