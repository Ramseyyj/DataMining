# dimensionality reduction
import numpy as np
import os
from scipy.sparse import csgraph

const_maxfloat = 1.7976931348623157e+308


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
def dimension_reduction(matrix, k):

    eig_val, eig_vec = np.linalg.eig(matrix)

    sorted_index = np.argsort(-eig_val, axis=1)

    eig_vec = eig_vec[sorted_index]

    return np.transpose(eig_vec[:k, :])


# PCA降维法
def PCA_dimension_reduction(matrix_train, matrix_test, k):

    # 把矩阵每一行的均值变为0
    matrix_train = matrix_mean_zero(matrix_train)

    # 求矩阵的协方差矩阵
    cov_matrix = np.cov(np.transpose(matrix_train))

    # 对协方差矩阵进行降维
    DR_matrix = dimension_reduction(cov_matrix, k)

    # 用训练集得到的降维矩阵对训练集和测试集进行降维
    DR_matrix_train = np.dot(matrix_train, DR_matrix)
    DR_matrix_test = np.dot(matrix_test, DR_matrix)

    return DR_matrix_train, DR_matrix_test


# SVD降维法
def SVD_dimension_reduction(matrix_train, matrix_test, k):

    u, s, v = np.linalg.svd(matrix_train.T)

    u = u[:, :k]

    DR_matrix_trian = np.dot(matrix_train, u)
    DR_matrix_test = np.dot(matrix_test, u)

    return DR_matrix_trian, DR_matrix_test


def KNN_algorithm(matrix, k_nn):

    sorted_index = np.argsort(matrix, axis=1)

    for i in range(matrix.shape[0]):

        k_min_index_set = set(sorted_index[i, :k_nn+1])

        for j in range(matrix.shape[1]):
            if j not in k_min_index_set:
                matrix[i, j] = 0.

    return matrix


def MDS_dimension_reduction(matrix, k):

    M = matrix.shape[0]

    matrix **= 2
    H = np.eye(M) - 1 / M
    B = -0.5 * np.dot(np.dot(H, matrix), H)

    eigVal, eigVec = np.linalg.eig(B)
    matrix = np.dot(eigVec[:, :k], np.diag(np.sqrt(eigVal[:k])))

    return matrix


def ISOMAP_dimension_reduction(matrix, k):

    graph = np.zeros((matrix.shape[0], matrix.shape[0]))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            graph[i, j] = np.linalg.norm(matrix[i, :] - matrix[j, :])

    graph = KNN_algorithm(graph, 21)
    graph = csgraph.shortest_path(graph, 'D', )

    matrix = MDS_dimension_reduction(graph, k)

    return matrix


# 判断精度计算
def accuracy_compute(matrix_train, matrix_test, labels_train, labels_test):

    matched_label_count = 0
    for i in range(matrix_test.shape[0]):
        min_dist = const_maxfloat
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

# SonarTrainDRMatrix, SonarTestDRMatrix = PCA_dimension_reduction(SonarTrainMatrix, SonarTestMatrix, 10)
# SpliceTrainDRMatrix, SpliceTestDRMatrix = PCA_dimension_reduction(SpliceTrainMatrix, SpliceTestMatrix, 10)
SonarTrainDRMatrix, SonarTestDRMatrix = SVD_dimension_reduction(SonarTrainMatrix, SonarTestMatrix, 30)
# SonarTrainDRMatrix = ISOMAP_dimension_reduction(SonarTrainMatrix, 10)
# SonarTestDRMatrix = ISOMAP_dimension_reduction(SonarTestMatrix, 10)

# print(SonarTrainDRMatrix)
# print(SonarTestDRMatrix)

Sonar_accuracy = accuracy_compute(SonarTrainDRMatrix, SonarTestDRMatrix, SonarTrainLabels, SonarTestLabels)
print(Sonar_accuracy)
# Splice_accuracy = accuracy_compute(SpliceTrainDRMatrix, SpliceTestDRMatrix, SpliceTrainLabels, SpliceTestLabels)
# print(Splice_accuracy)

