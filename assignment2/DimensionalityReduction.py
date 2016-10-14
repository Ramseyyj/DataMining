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


def accuracy_compute(matrix_train, matrix_test, labels_train, labels_test):

    matched_label_count = 0
    for i in range(matrix_test.shape[0]):
        min_dist = 0
        min_index = 0
        for j in range(matrix_train.shape[0]):

            dist = np.linalg.norm(matrix_test[i, :] - matrix_train[j, :])

            if dist < min_dist:
                min_dist = dist
                min_index = j

        if labels_test[i] == labels_train[min_index]:
            matched_label_count += 1

    accuracy = matched_label_count / matrix_test.shape[0]
    return accuracy


# 把矩阵从txt提取出来并把最后一列标志位进行分离
SonarTrainMatrix, SonarTrainLabels = load_matrix_from_txt('sonar-train.txt')
SonarTestMatrix, SonarTestLabels = load_matrix_from_txt('sonar-test.txt')
SpliceTrainMatrix, SpliceTrainLabels = load_matrix_from_txt('splice-train.txt')
SpliceTestMatrix, SpliceTestLabels = load_matrix_from_txt('splice-test.txt')


# 把矩阵每一行的均值变为0
SonarTrainMatrix = matrix_mean_zero(SonarTrainMatrix)
# SonarTestMatrix = matrix_mean_zero(SonarTestMatrix)
SpliceTrainMatrix = matrix_mean_zero(SpliceTrainMatrix)
# SpliceTestMatrix = matrix_mean_zero(SpliceTestMatrix)


# 求矩阵的协方差矩阵
SonarTrainCovMatrix = np.cov(np.transpose(SonarTrainMatrix))
# SonarTestCovMatrix = np.cov(np.transpose(SonarTestMatrix))
SpliceTrainCovMatrix = np.cov(np.transpose(SpliceTrainMatrix))
# SpliceTestCovMatrix = np.cov(np.transpose(SpliceTestMatrix))

Sonar_DRMatrix = dimensionality_reduction(SonarTrainCovMatrix, 10)
Splice_DRMatrix = dimensionality_reduction(SpliceTrainCovMatrix, 10)

SonarTrainDRMatrix = np.dot(SonarTrainMatrix, Sonar_DRMatrix)
SonarTestDRMatrix = np.dot(SonarTestMatrix, Sonar_DRMatrix)
SpliceTrainDRMatrix = np.dot(SpliceTrainMatrix, Splice_DRMatrix)
SpliceTestDRMatrix = np.dot(SpliceTestMatrix, Splice_DRMatrix)

Sonar_accuracy = accuracy_compute(SonarTrainDRMatrix, SonarTestDRMatrix, SonarTrainLabels, SonarTestLabels)
print(Sonar_accuracy)
Splice_accuracy = accuracy_compute(SpliceTrainDRMatrix, SpliceTestDRMatrix, SpliceTrainLabels, SpliceTestLabels)
print(Splice_accuracy)


