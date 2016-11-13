import numpy as np
import os


def load_matrix_label_from_txt(filename):

    root_dir = os.path.abspath('.') + '\\data'
    txt_file_dir = root_dir + '\\' + filename

    matrix_origin = np.loadtxt(txt_file_dir)
    column_count = matrix_origin.shape[1]

    return matrix_origin[:, :column_count-1], matrix_origin[:, column_count]


dataset1_train_txt = 'dataset1-a9a-training.txt'
dataset1_test_txt = 'dataset1-a9a-testing.txt'
dataset2_train_txt = 'covtype-training.txt'
dataset2_test_txt = 'covtype-testing.txt'

