# Clustering

import numpy as np
import os
import random
import datetime
import pandas as pd

const_maxfloat = 1.7976931348623157e+308


# 从txt文件中读取矩阵
def load_matrix_from_txt(txt_filename, label_flag):

    root_dir = os.path.abspath('.') + '\\data'
    txt_file_dir = root_dir + '\\' + txt_filename

    if label_flag:
        lines = open(txt_file_dir).readlines()
        fp = open(root_dir + '\\modify_' + txt_filename, 'w')

        for line in lines:
            fp.write(line.replace(',', ' '))
        fp.close()

        matrix = np.loadtxt(root_dir + '\\modify_' + txt_filename)

        labels = []
        column_count = matrix.shape[1]
        for row in matrix:
            labels.append(row[column_count-1])

        matrix = matrix[:, :column_count-1]

        return matrix, labels
    else:
        matrix = np.loadtxt(txt_file_dir)

        return matrix


def random_medoids(matrix, k):

    medoids = {}

    for i in range(k):
        index = random.randint(0, matrix.shape[0]-1)
        medoids[i] = [index, matrix[index, :], set([])]

    # for i in range(k):
    #     medoids[i] = [i, matrix[i, :], set([])]

    return medoids


def init_medoids(matrix, k):

    medoids = {}

    for i in range(k):
        medoids[i] = [0, matrix[0, :], set([])]

    return medoids


def compare_medoids(medoids1, medoids2):

    for i in range(len(medoids1)):
        if medoids1[i] != medoids2[i][0]:
            return False

    return True


def min_medoid_index(medoids, vector):

    min_dist = const_maxfloat
    min_index = 0

    for i in range(len(medoids)):
        dist = np.linalg.norm(vector - medoids[i][1])

        if dist < min_dist:
            min_dist = dist
            min_index = i

    return min_index


def clustering(medoids, matrix):

    for i in range(len(medoids)):
        medoids[i][2].clear()

    for i in range(matrix.shape[0]):
        medoids[min_medoid_index(medoids, matrix[i, :])][2].add(i)

    return medoids


def k_medoid_clustering(matrix, k):

    medoids = random_medoids(matrix, k)

    temp_medoids = [0 * i for i in range(k)]

    count = 0

    while not compare_medoids(temp_medoids, medoids):

        count += 1

        for i in range(k):
            temp_medoids[i] = medoids[i][0]

        medoids = clustering(medoids, matrix)

        for i in range(len(medoids)):

            min_dist = const_maxfloat
            min_index = 0
            for j in medoids[i][2]:

                sum_dist = 0
                for s in medoids[i][2]:
                    sum_dist += np.sum(abs(matrix[j, :] - matrix[s, :]))

                if sum_dist < min_dist:
                    min_dist = sum_dist
                    min_index = j

            medoids[i][0] = min_index
            medoids[i][1] = matrix[min_index, :]

    print('k-medoid迭代次数：%d' % count)

    return medoids


def numbers_of_cluster_from_classes(labels, medoids_result, label_names, k):

    m = np.zeros((k, k))

    for i in range(k):
        for j in range(k):

            count = 0

            for s in medoids_result[j][2]:
                if labels[s] == label_names[i]:
                    count += 1

            m[i, j] = count

    return m


def graph_generating(matrix, filename):

    adj_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            adj_matrix[i, j] = np.sum(abs(matrix[i] - matrix[j]))

    np.savetxt(filename, adj_matrix, fmt='%.5f', newline='\n')


def laplaction_eigenmap_dimension_reduction(matrix, k):

    eigval, eigvec = np.linalg.eig(matrix)

    k_smallest_index = np.argsort(eigval)[0:k]

    return eigvec[:, k_smallest_index]


def spectral_clustering(filename, k_nn, k):

    adj_matrix = pd.read_csv(filename, delimiter=' ', header=None).values

    W = np.zeros(adj_matrix.shape)

    for i in range(adj_matrix.shape[0]):

        sorted_index = np.argsort(adj_matrix[i])

        k_min_index_set = set(sorted_index[:k_nn + 1])

        for j in range(adj_matrix.shape[1]):
            if j in k_min_index_set:
                W[i, j] = 1

    d = np.sum(W, axis=0)

    D = np.diag(d)

    L = D - W

    Lk = laplaction_eigenmap_dimension_reduction(L, k)

    return k_medoid_clustering(Lk, k)


def purity_compute(m, k):

    tatol = 0

    for i in range(k):
        for j in range(k):
            tatol += m[i, j]

    Pj_sum = 0
    for i in range(k):
        Pj_sum += m[:, i].max()

    return Pj_sum / tatol


def Gini_index_compute(m, k):

    M = np.sum(m, axis=0)

    G = np.zeros(k, dtype=np.float64)
    for j in range(k):

        temp_sum = 0
        for i in range(k):
            temp_sum += (m[i, j] / M[j])**2

        G[j] = 1 - temp_sum

    sum1 = 0
    for j in range(k):
        sum1 += G[j]*M[j]

    sum2 = 0
    for j in range(k):
        sum2 += M[j]

    return sum1 / sum2


def evaluation(labels, cluster, label_names):

    Mij = numbers_of_cluster_from_classes(labels, cluster, label_names, k_value)

    Purity = purity_compute(Mij, k_value)
    GiniIndex = Gini_index_compute(Mij, k_value)
    print('Purity：%f' % Purity)
    print('Gini Index：%f' % GiniIndex)


def main():

    starttime = datetime.datetime.now()

    # dist_matrix_filename = 'dist_matrix_german.txt'
    # german_matrix, german_labels = load_matrix_from_txt('german.txt', True)

    dist_matrix_filename = 'dist_matrix.txt'
    german_matrix, german_labels = load_matrix_from_txt('mnist.txt', True)

    # graph_generating(german_matrix, dist_matrix_filename)

    LabelNames = []

    if k_value == 2:
        LabelNames = [1, -1]
    elif k_value == 10:
        LabelNames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        print('error!')
        exit()

    time = [datetime.datetime.now() for i in range(4)]
    print('k_medoids:')
    clustering_result = k_medoid_clustering(german_matrix, k_value)
    evaluation(german_labels, clustering_result, LabelNames)
    time[0] = datetime.datetime.now()
    print('运行时间：%d 秒' % (time[0] - starttime).seconds)

    print('spectral_cluster:')
    for K_nn in range(3, 10, 3):
        clustering_result = spectral_clustering(dist_matrix_filename, K_nn, k_value)
        print('K_nn=%d' % K_nn)
        evaluation(german_labels, clustering_result, LabelNames)
        time[int(K_nn / 3)] = datetime.datetime.now()
        print('运行时间：%d 秒' % (time[int(K_nn / 3)] - time[int(K_nn / 3 - 1)]).seconds)


if __name__ == "__main__":

    k_value = 2
    main()
