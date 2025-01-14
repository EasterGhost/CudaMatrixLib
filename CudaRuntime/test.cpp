#include "cuda_matrix.h"

int main() {
    // 初始化一个 5x5 的矩阵
    int rows = 5;
    int cols = 5;
    cudaMatrix matrix(rows, cols);

    // 填充矩阵数据
    vector<float> hostData(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            hostData[i * cols + j] = static_cast<float>(i * cols + j);
        }
    }
    matrix.setData(hostData);

    // 打印原始矩阵
    cout << "Original Matrix:" << endl;
    matrix.printData();

    // 定义行索引和列索引
    vector<int> row_indices = { 1, 3, 4 }; // 提取第 1、3、4 行
    vector<int> col_indices = { 0, 2 };    // 提取第 0、2 列

    // 提取子矩阵
    cudaMatrix subMatrix = matrix.subMatrix(row_indices, col_indices);

    // 打印子矩阵
    cout << "Sub Matrix:" << endl;
    subMatrix.printData();
	system("pause");
    return 0;
}
