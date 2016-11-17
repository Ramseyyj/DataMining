import numpy as np
import math
import os


def load_data_from_txt(filename):

    root_dir = os.path.abspath('.') + '\\data'
    txt_file_dir = root_dir + '\\' + filename

    f_read = open(txt_file_dir, 'r')
    f_write = open('matrix_'+filename, 'w')

    index = 0
    for line in f_read:
        if index == 0:
            feature_category = line.strip('\n').split(',')
        else:
            f_write.writelines(line)

        index += 1

    f_read.close()
    f_write.close()

    matrix_origin = np.loadtxt('matrix_'+filename, delimiter=',')
    column_count = matrix_origin.shape[1]

    return matrix_origin[:, :column_count - 1], matrix_origin[:, column_count - 1], feature_category


def feature_conditional_probability(matrix, label, feature_category):

    label_dict = {}
    for i in range(len(label)):
        if label[i] in label_dict.keys():
            label_dict[label[i]] += 1
        else:
            label_dict[label[i]] = 1

    feature_conditional_probability_dict = {}
    for i in range(matrix.shape[1]):

        if feature_category == 1:

            for j in range(matrix.shape[0]):
                if (matrix[j, i], label[j]) in feature_conditional_probability_dict.keys():
                    feature_conditional_probability_dict[(matrix[j, i], label[j])] += 1 / label_dict[label[j]]
                else:
                    feature_conditional_probability_dict[(matrix[j, i], label[j])] = 1 / label_dict[label[j]]

        elif feature_category == 0:

            average = np.mean(matrix[:, i])
            variance = np.var(matrix[:, i])

        else:
            print('error!')
            exit()






dataset1_filename = 'breast-cancer-assignment5.txt'
dataset2_filename = 'german-assignment5.txt'

dataset1_matrix, dataset1_label, dataset1_feature_category = load_data_from_txt(dataset1_filename)


