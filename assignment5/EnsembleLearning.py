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

        if feature_category[i] == '1':

            for j in range(matrix.shape[0]):
                if (matrix[j, i], label[j]) in feature_conditional_probability_dict.keys():
                    feature_conditional_probability_dict[(matrix[j, i], label[j])] += 1 / label_dict[label[j]]
                else:
                    feature_conditional_probability_dict[(matrix[j, i], label[j])] = 1 / label_dict[label[j]]

        elif feature_category[i] == '0':

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


def error_ratio(label, classifier_result, weight_list):

    result = 0

    for i in range(len(label)):
        if label[i] != classifier_result[i]:
            result += weight_list[i]

    return result


def adaboost_core(matrix, label, feature_category, Iteration_count):

    weight_list = list()
    for i in range(matrix.shape[0]):
        weight_list.append(1 / matrix.shape[0])

    classifier_coreDict_list = list()
    at_list = list()

    for t in range(Iteration_count):
        temp_dict = feature_conditional_probability(matrix, label, feature_category)

        classifier_result = list()
        for i in range(matrix.shape[0]):
            classifier_result.append(naive_bayes_classifier(temp_dict, matrix[i]))

        et = error_ratio(label, classifier_result, weight_list)

        if et > 0.5:
            break

        classifier_coreDict_list.append(temp_dict)

        at = 0.5 * math.log((1 - et) / et)
        at_list.append(at)

        Zt = 0
        for i in range(matrix.shape[0]):
            Zt += weight_list[i]
            weight_list[i] *= math.exp(-at*label[i]*classifier_result[i])

        for i in range(matrix.shape[0]):
            weight_list[i] /= Zt

    return classifier_coreDict_list, at_list


def adaboost_algorithm(matrix_train, matrix_test, label_train, feature_category, Iteration_count):

    classifier_coreDict_list, at_list = adaboost_core(matrix_train, label_train, feature_category, Iteration_count)

    result_list = list()

    for i in range(matrix_test.shape[0]):
        temp_sum = 0
        for t in range(len(at_list)):
            temp_sum += at_list[t] * naive_bayes_classifier(classifier_coreDict_list[t], matrix_test[i])

        temp_result = np.sign(temp_sum)
        result_list.append(temp_result)

    return result_list


def accuracy_compute(label, result_list):

    count = 0

    for i in range(len(label)):
        if label[i] == result_list[i]:
            count += 1

    return count / len(label)


def fold_cross_validation(fold_count, matrix, label, feature_category, Iteration_count):

    each_count = int(matrix.shape[0] / fold_count) + 1

    accuracy_list = list()

    for i in range(fold_count):
        if i == fold_count - 1:
            min_index = 0 + i*each_count

            matrix_train = matrix[:min_index]
            matrix_test = matrix[min_index:]
            label_train = label[:min_index]
            label_test = label[min_index:]
        else:
            min_index = 0 + i * each_count
            max_index = 9 + i * each_count
            train_index_list = list()
            test_index_list = list()
            label_train = list()
            label_test = list()

            for j in range(matrix.shape[0]):
                if min_index <= j <= max_index:
                    test_index_list.append(j)
                    label_test.append(label[j])
                else:
                    train_index_list.append(j)
                    label_train.append(label[j])

            matrix_train = matrix[train_index_list]
            matrix_test = matrix[test_index_list]

        result_list = adaboost_algorithm(matrix_train, matrix_test, label_train, feature_category, Iteration_count)

        accuracy_list.append(accuracy_compute(label_test, result_list))

    result_mean = np.mean(accuracy_list)
    result_standard_deviation = np.std(accuracy_list, ddof=1)

    print('%s, %s' % (result_mean, result_standard_deviation))


dataset1_filename = 'breast-cancer-assignment5.txt'
dataset2_filename = 'german-assignment5.txt'

dataset1_matrix, dataset1_label, dataset1_feature_category = load_data_from_txt(dataset1_filename)

# 把数据集1的标签0改成-1，对标签进行统一，都变成-1和1
for index_g in range(len(dataset1_label)):
    if dataset1_label[index_g] == 0:
        dataset1_label[index_g] = -1

fold_cross_validation(10, dataset1_matrix, dataset1_label, dataset1_feature_category, 10)

