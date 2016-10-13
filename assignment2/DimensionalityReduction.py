# implement PCA for dimensionality reduction

import numpy as np
import os


def load_matrix_from_txt(txt_filename):

    root_dir = os.path.abspath('.') + '\\data'
    txt_file_dir = root_dir + '\\' + txt_filename

    lines = open(txt_file_dir).readlines()
    fp = open(root_dir+'\\modify_'+txt_filename, 'w')

    for line in lines:
        fp.write(line.replace(',', ' '))
    fp.close()

    matrix = np.loadtxt(root_dir+'\\modify_'+txt_filename)

    labels = []
    column_count = matrix.shape[1]
    for row in matrix:
        labels.append(row[column_count-1])

    matrix = matrix[:, :column_count-1]

    return matrix, labels


def matrix_mean_zero(matrix):

    a = np.mean(matrix, axis=1)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            matrix[row, column] -= a[row]

    return matrix


def dimensionality_reduction(matrix, k):

    eig_val, eig_vec = np.linalg.eig(matrix)

    sorted_index = np.argsort(-eig_val, axis=0)

    eig_vec = eig_vec[sorted_index]

    return np.transpose(eig_vec[:k, :])


# 把矩阵从txt提取出来并把最后一列标志位进行分离
SonarTestMatrix, SonarTestLabels = load_matrix_from_txt('sonar-test.txt')
# SonarTrainMatrix, SonarTrainLabels = load_matrix_from_txt('sonar-train.txt')
# SpliceTestMatrix, SpliceTestLabels = load_matrix_from_txt('splice-test.txt')
# SpliceTrainMatrix, SpliceTrainLabels = load_matrix_from_txt('splice-train.txt')

# print(SonarTestMatrix)
# print(SonarTestLabels)
# print(SonarTrainMatrix)
# print(SonarTestLabels)
# print(SpliceTestMatrix)
# print(SpliceTestLabels)
# print(SpliceTrainMatrix)
# print(SpliceTrainLabels)

# 把矩阵每一行的均值变为0
SonarTestMatrix = matrix_mean_zero(SonarTestMatrix)
# SonarTrainMatrix = matrix_mean_zero(SonarTrainMatrix)
# SpliceTestMatrix = matrix_mean_zero(SpliceTestMatrix)
# SpliceTrainMatrix = matrix_mean_zero(SpliceTrainMatrix)

# 求矩阵的协方差矩阵
SonarTestCovMatrix = np.cov(np.transpose(SonarTestMatrix))
# SonarTrainCovMatrix = np.cov(np.transpose(SonarTrainMatrix))
# SpliceTestCovMatrix = np.cov(np.transpose(SpliceTestMatrix))
# SpliceTrainCovMatrix = np.cov(np.transpose(SpliceTrainMatrix))

SonarTestDRMatrix = dimensionality_reduction(SonarTestCovMatrix, 5)

print(np.dot(SonarTestMatrix, SonarTestDRMatrix))
print(np.dot(SonarTestMatrix, SonarTestDRMatrix).shape)


