# dimensionality reduction
import numpy as np
import os


# 从txt文件中读取矩阵
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


# 把矩阵每一行的均值变为0
def matrix_mean_zero(matrix):

    a = np.mean(matrix, axis=1)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            matrix[row, column] -= a[row]

    return matrix


# 根据协方差矩阵得到降维矩阵
def dimensionality_reduction(matrix, k):

    eig_val, eig_vec = np.linalg.eig(matrix)

    sorted_index = np.argsort(-eig_val, axis=0)

    eig_vec = eig_vec[sorted_index]

    return np.transpose(eig_vec[:k, :])


# PCA 降维法
def PCA_dimensionality_reduction(matrix_train, matrix_test, k):

    # 把矩阵每一行的均值变为0
    matrix_train = matrix_mean_zero(matrix_train)

    # 求矩阵的协方差矩阵
    cov_matrix = np.cov(np.transpose(matrix_train))

    # 对协方差矩阵进行降维
    DR_matrix = dimensionality_reduction(cov_matrix, k)

    # 用训练集得到的降维矩阵对训练集和测试集进行降维
    DR_matrix_train = np.dot(matrix_train, DR_matrix)
    DR_matrix_test = np.dot(matrix_test, DR_matrix)

    return DR_matrix_train, DR_matrix_test


# 判断精度计算
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

SonarTrainDRMatrix, SonarTestDRMatrix = PCA_dimensionality_reduction(SonarTrainMatrix, SonarTestMatrix, 10)
SpliceTrainDRMatrix, SpliceTestDRMatrix = PCA_dimensionality_reduction(SpliceTrainMatrix, SpliceTestMatrix, 10)

Sonar_accuracy = accuracy_compute(SonarTrainDRMatrix, SonarTestDRMatrix, SonarTrainLabels, SonarTestLabels)
print(Sonar_accuracy)
Splice_accuracy = accuracy_compute(SpliceTrainDRMatrix, SpliceTestDRMatrix, SpliceTrainLabels, SpliceTestLabels)
print(Splice_accuracy)

