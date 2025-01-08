#include "cuda_matrix.h"
#include "cuda_matrix.cu"
int main() {
	try {
		// ����һ�� 3x3 ��ȫ�����
		cudaMatrix<float> mat1(3, 3, Zero);
		std::cout << "Matrix 1 (Zero Matrix):" << std::endl;
		mat1.printData();

		// ����һ�� 3x3 ��ȫһ����
		cudaMatrix<float> mat2(3, 3, Ones);
		std::cout << "Matrix 2 (Ones Matrix):" << std::endl;
		mat2.printData();

		// ����һ�� 3x3 �ĵ�λ����
		cudaMatrix<float> mat3(3, 3, Identity);
		std::cout << "Matrix 3 (Identity Matrix):" << std::endl;
		mat3.printData();

		// ����һ�� 3x3 ���������
		cudaMatrix<float> mat4(3, 3, Random);
		std::cout << "Matrix 4 (Random Matrix):" << std::endl;
		mat4.printData();

		// ����ӷ�
		cudaMatrix<float> mat5 = mat2 + mat3;
		std::cout << "Matrix 5 (Matrix 2 + Matrix 3):" << std::endl;
		mat5.printData();

		// �������
		cudaMatrix<float> mat6 = mat2 - mat3;
		std::cout << "Matrix 6 (Matrix 2 - Matrix 3):" << std::endl;
		mat6.printData();

		// ����˷�
		cudaMatrix<float> mat7 = mat2 * mat3;
		std::cout << "Matrix 7 (Matrix 2 * Matrix 3):" << std::endl;
		mat7.printData();

		// ����ת��
		cudaMatrix<float> mat8 = ~mat4;
		std::cout << "Matrix 8 (Transpose of Matrix 4):" << std::endl;
		mat8.printData();

		// ������
		float trace = mat3.trace();
		std::cout << "Trace of Matrix 3: " << trace << std::endl;

		// ��������
		float norm = mat4.norm(2);
		std::cout << "Norm of Matrix 4: " << norm << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
	}
	system("pause");
	return 0;
}