// �����ļ���test_assemble_blocks.cpp

#include "cuda_matrix.h"

int main() {
	// �����Ӿ��� A, B, C, D
	cudaMatrix A(2, 2);
	A.setData({ 1, 2, 3, 4 });

	cudaMatrix B(2, 3);
	B.setData({ 5, 6, 7, 8, 9, 10 });

	cudaMatrix C(3, 2);
	C.setData({ 11, 12, 13, 14, 15, 16 });

	cudaMatrix D(3, 3);
	D.setData({ 17, 18, 19, 20, 21, 22, 23, 24, 25 });

	// ���� O �������磬ȫ�����
	//cudaMatrix O = cudaMatrix::zeros(2, 2);

	// ���Ӿ������������
	std::vector<std::vector<cudaMatrix>> blocks = {
		{A, B},
		{C, D}
	};

	// ʹ�� assembleBlocks ������Ͼ���
	cudaMatrix assembledMatrix = cudaMatrix::assembleBlocks(blocks);

	// ��ӡ�������
	std::cout << "Assembled Matrix:" << std::endl;
	assembledMatrix.printData();

	// ʹ�� O ������Ϊռλ��
	std::vector<std::vector<cudaMatrix>> blocksWithO = {
		{A, O},
		{O, D}
	};

	// ������Ͼ���
	cudaMatrix assembledMatrixWithO = cudaMatrix::assembleBlocks(blocksWithO);

	// ��ӡ���� O ����Ľ��
	std::cout << "\nAssembled Matrix with O Matrix:" << std::endl;
	assembledMatrixWithO.printData();
	system("pause");
	return 0;
}