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


def run_some_test_on_data(file):

    root_dir = os.path.abspath('.') + '\\data'
    csv_file_dir = root_dir + '\\' + file

    line_number_count = 0
    customer_set = set()
    zero_new_product_count = 0
    one_new_product_count = 0
    other_new_product_count = 0

    with open(csv_file_dir, 'r', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        for row in spamreader:

            line_number_count += 1
            if line_number_count % 100000 == 0:
                print(line_number_count / 100000)

            settlement_date = row[0]
            customer_code = row[1]

            temp_count = 0
            for i in range(24, 48, 1):
                if row[i] == '1':
                    temp_count += 1

            # if temp_count == 0:
            #     zero_new_product_count += 1
            # elif temp_count == 1:
            #     one_new_product_count += 1
            # else:
            #     other_new_product_count += 1

            if settlement_date == '2015-06-28' and temp_count == 1:
                customer_set.add(customer_code)

    print('line_number_count: %d' % line_number_count)
    print('customer_set count: %d' % len(customer_set))
    print('zero_new_product_count: %d' % zero_new_product_count)
    print('one_new_product_count: %d' % one_new_product_count)
    print('other_new_product_count: %d' % other_new_product_count)

# reduce_dataset(test_file, 'test_temp', 100)
print_csv_in_lines('train_temp', 5)
print_csv_in_lines('test_temp', 5)

# run_some_test_on_data(train_file)
