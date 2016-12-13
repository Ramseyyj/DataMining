import csv
import datetime
import os


train_file = 'train_ver2.csv'
test_file = 'test_ver2.csv'


def print_csv_in_lines(file, lines=0):

    with open(file, 'r', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        if lines != 0:
            testline = lines
            for row in spamreader:
                print(row)
                testline -= 1
                if testline == 0:
                    break
        else:
            for row in spamreader:
                print(row)


def reduce_dataset(origin_file, reduce_file, reduce_lines):

    root_dir = os.path.abspath('.') + '\\data'
    csv_file_dir = root_dir + '\\' + origin_file

    with open(csv_file_dir, 'r', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        with open(reduce_file, 'w', encoding='utf-8', newline='\n') as csvwritefile:
            csvwriter = csv.writer(csvwritefile, delimiter=',')
            testline = reduce_lines
            for row in spamreader:
                csvwriter.writerow(row)
                testline -= 1
                if testline == 0:
                    break

reduce_dataset(train_file, 'train_temp', 100)
print_csv_in_lines('train_temp', 10)
