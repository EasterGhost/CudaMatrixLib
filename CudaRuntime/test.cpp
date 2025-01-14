#include "cuda_matrix.h"

int main() {
    // ��ʼ��һ�� 5x5 �ľ���
    int rows = 5;
    int cols = 5;
    cudaMatrix matrix(rows, cols);

    // ����������
    vector<float> hostData(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            hostData[i * cols + j] = static_cast<float>(i * cols + j);
        }
    }
    matrix.setData(hostData);

    // ��ӡԭʼ����
    cout << "Original Matrix:" << endl;
    matrix.printData();

    // ������������������
    vector<int> row_indices = { 1, 3, 4 }; // ��ȡ�� 1��3��4 ��
    vector<int> col_indices = { 0, 2 };    // ��ȡ�� 0��2 ��

    // ��ȡ�Ӿ���
    cudaMatrix subMatrix = matrix.subMatrix(row_indices, col_indices);

    // ��ӡ�Ӿ���
    cout << "Sub Matrix:" << endl;
    subMatrix.printData();
	system("pause");
    return 0;
}
