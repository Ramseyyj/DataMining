# Clustering

import numpy as np
import os
import random

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
                    sum_dist += np.linalg.norm(matrix[j, :] - matrix[s, :])

                if sum_dist < min_dist:
                    min_dist = sum_dist
                    min_index = j

            medoids[i][0] = min_index
            medoids[i][1] = matrix[min_index, :]

    print(count)

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

    G = np.arange(k)
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


# german_matrix, german_labels = load_matrix_from_txt('german.txt', True)
german_matrix, german_labels = load_matrix_from_txt('mnist.txt', True)
clustering_result = k_medoid_clustering(german_matrix, 10)
# print(clustering_result)

# LabelNames = [1, -1]
LabelNames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Mij = numbers_of_cluster_from_classes(german_labels, clustering_result, LabelNames, 10)
# print(Mij)
Purity = purity_compute(Mij, 10)
GiniIndex = Gini_index_compute(Mij, 10)
print(Purity)
print(GiniIndex)

