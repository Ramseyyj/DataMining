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

    feature_conditional_probability_dict = dict()

    feature_conditional_probability_dict[-1] = label_dict[-1] / len(label)
    feature_conditional_probability_dict[1] = label_dict[1] / len(label)

    for i in range(matrix.shape[1]):

        if feature_category == 1:

            for j in range(matrix.shape[0]):
                if (matrix[j, i], label[j]) in feature_conditional_probability_dict.keys():
                    feature_conditional_probability_dict[(matrix[j, i], label[j])] += 1 / label_dict[label[j]]
                else:
                    feature_conditional_probability_dict[(matrix[j, i], label[j])] = 1 / label_dict[label[j]]

        elif feature_category == 0:

            category_dict = {}
            for j in range(matrix.shape[0]):
                if label[j] in category_dict.keys():
                    category_dict[label[j]].append(matrix[j, i])
                else:
                    category_dict[label[j]] = []
                    category_dict[label[j]].append(matrix[j, i])

            for j in range(matrix.shape[0]):
                average = np.mean(category_dict[label[j]])
                variance = np.var(category_dict[label[j]])
                feature_conditional_probability_dict[(matrix[j, i], label[j])] \
                    = math.exp(-(matrix[j, i] - variance)**2 / (2*average**2)) / (math.sqrt(2*math.pi)*average)

        else:
            print('error!')
            exit()

    return feature_conditional_probability_dict


def naive_bayes_classifier(feature_conditional_probability_dict, vector):

    probability_label_negative_one = feature_conditional_probability_dict[-1]
    probability_label_postive_one = feature_conditional_probability_dict[1]

    for i in range(vector.shape[0]):
        probability_label_negative_one *= feature_conditional_probability_dict[(vector[i], -1)]
        probability_label_postive_one *= feature_conditional_probability_dict[(vector[i], 1)]

    if probability_label_postive_one > probability_label_negative_one:
        return 1
    else:
        return -1


dataset1_filename = 'breast-cancer-assignment5.txt'
dataset2_filename = 'german-assignment5.txt'

dataset1_matrix, dataset1_label, dataset1_feature_category = load_data_from_txt(dataset1_filename)

# 把数据集1的标签0改成-1，对标签进行统一，都变成-1和1
for index_g in range(len(dataset1_label)):
    if dataset1_label[index_g] == 0:
        dataset1_label[index_g] = -1

