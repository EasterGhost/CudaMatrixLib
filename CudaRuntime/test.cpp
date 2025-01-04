// 测试文件：test_assemble_blocks.cpp

#include "cuda_matrix.h"

int main() {
	// 创建子矩阵 A, B, C, D
	cudaMatrix A(2, 2);
	A.setData({ 1, 2, 3, 4 });

	cudaMatrix B(2, 3);
	B.setData({ 5, 6, 7, 8, 9, 10 });

	cudaMatrix C(3, 2);
	C.setData({ 11, 12, 13, 14, 15, 16 });

	cudaMatrix D(3, 3);
	D.setData({ 17, 18, 19, 20, 21, 22, 23, 24, 25 });

	// 定义 O 矩阵（例如，全零矩阵）
	//cudaMatrix O = cudaMatrix::zeros(2, 2);

	// 将子矩阵放入块矩阵中
	std::vector<std::vector<cudaMatrix>> blocks = {
		{A, B},
		{C, D}
	};

	// 使用 assembleBlocks 方法组合矩阵
	cudaMatrix assembledMatrix = cudaMatrix::assembleBlocks(blocks);

	// 打印结果矩阵
	std::cout << "Assembled Matrix:" << std::endl;
	assembledMatrix.printData();

	// 使用 O 矩阵作为占位符
	std::vector<std::vector<cudaMatrix>> blocksWithO = {
		{A, O},
		{O, D}
	};

	// 重新组合矩阵
	cudaMatrix assembledMatrixWithO = cudaMatrix::assembleBlocks(blocksWithO);

	// 打印包含 O 矩阵的结果
	std::cout << "\nAssembled Matrix with O Matrix:" << std::endl;
	assembledMatrixWithO.printData();
	system("pause");
	return 0;
}