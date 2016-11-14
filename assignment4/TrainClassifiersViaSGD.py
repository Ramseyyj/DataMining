import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt

const_lambda = 1
const_gamma_step = 2
const_output_step = 3000


def random_vector(dimension, min_value, max_value):
    vector = np.zeros(dimension)

    for i in range(dimension):
        vector[i] = random.uniform(min_value, max_value)
    return vector


def random_list(number):

    init_list = [i for i in range(number)]
    result_list = []

    for i in range(number):
        temp_index = random.randint(1, number-i) - 1
        result_list.append(init_list[temp_index])
        init_list.pop(temp_index)

    return result_list


def load_matrix_label_from_txt(filename):

    root_dir = os.path.abspath('.') + '\\data'
    txt_file_dir = root_dir + '\\' + filename

    matrix_origin = np.loadtxt(txt_file_dir, delimiter=',')
    column_count = matrix_origin.shape[1]

    return matrix_origin[:, :column_count-1], matrix_origin[:, column_count-1]


def norm_one_partial_derivative(vector):
    for i in range(vector.shape[0]):
        if vector[i] >= 0:
            vector[i] = 1
        else:
            vector[i] = -1
    return vector


def function_a(vector_beta, vector_xi, yi):

    temp = np.dot(vector_beta.T, vector_xi)

    temp = math.exp(-yi * temp) * -yi / (1 + math.exp(-yi * temp))

    vector_alpha = temp*vector_xi + const_lambda*norm_one_partial_derivative(vector_beta)

    return vector_alpha


def iteration_beta(vector_beta, vector_alpha):

    vector_beta -= const_gamma_step*vector_alpha

    return vector_beta


def error_ratio_compute(vector_beta, matrix, label):

    error_count = 0

    for i in range(matrix.shape[0]):
        temp = np.dot(vector_beta.T, matrix[i])
        if temp > 0:
            temp_label = 1
        else:
            temp_label = -1

        if temp_label != label[i]:
            error_count += 1

    return float(error_count) / matrix.shape[0]


def SGD(matrix_train, matrix_test, label_train, label_test, iteration_count):

    train_error_ratio_list = []
    test_error_ratio_list = []
    beta = random_vector(matrix_train.shape[1], 0, 1)
    error_ratio_count = 0
    iteration_beta_count = 0

    for i in range(iteration_count):

        temp_list = random_list(matrix_train.shape[0])

        for j in range(matrix_train.shape[0]):

            index = temp_list[j]
            alpha = function_a(beta, matrix_train[index], label_train[index])
            beta = iteration_beta(beta, alpha)
            iteration_beta_count += 1

            if iteration_beta_count == const_output_step:
                train_error_ratio_list.append(error_ratio_compute(beta, matrix_train, label_train))
                test_error_ratio_list.append(error_ratio_compute(beta, matrix_test, label_test))
                iteration_beta_count = 0
                error_ratio_count += 1

    x = [i for i in range(error_ratio_count)]

    plt.plot(x, train_error_ratio_list)
    plt.show()

dataset1_train_txt = 'dataset1-a9a-training.txt'
dataset1_test_txt = 'dataset1-a9a-testing.txt'
dataset2_train_txt = 'covtype-training.txt'
dataset2_test_txt = 'covtype-testing.txt'

dataset1_train_matrix, dataset1_train_label = load_matrix_label_from_txt(dataset1_train_txt)
dataset1_test_matrix, dataset1_test_label = load_matrix_label_from_txt(dataset1_test_txt)

SGD(dataset1_train_matrix, dataset1_test_matrix, dataset1_train_label, dataset1_test_label, 10)

