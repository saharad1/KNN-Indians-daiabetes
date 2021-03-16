from sys import argv
import os
from cross_validation import CrossValidation
from knn import KNN
from metrics import accuracy_score
from normalization import *


def load_data():
    """
    Loads data from path in first argument
    :return: returns data as list of Point
    """
    if len(argv) < 2:
        print('Not enough arguments provided. Please provide the path to the input file')
        exit(1)
    input_path = argv[1]

    if not os.path.exists(input_path):
        print('Input file does not exist')
        exit(1)

    points = []
    with open(input_path, 'r') as f:
        for index, row in enumerate(f.readlines()):
            row = row.strip()
            values = row.split(',')
            points.append(Point(str(index), values[:-1], values[-1]))
    return points


def run_knn(points):
    print("Question 3:")
    print("K=7")
    num_of_folds = [2, 10, 20]
    for fold_n in num_of_folds:
        print("{}-fold-cross-validation:".format(fold_n))
        m = KNN(19)
        m.train(points)
        cv = CrossValidation()
        cv.run_cv(points, fold_n, m, accuracy_score, False, True)

    print("Question 4:")
    for k_num in [5, 7]:
        print("K=" + str(k_num))

        # The Dummy Normalizer phase
        m2 = KNN(k_num)
        m2.train(points)
        cv = CrossValidation()
        name_dummy_normalizer = cv.run_cv(points, 2, m2, accuracy_score, False, True)
        print("Accuracy of DummyNormalizer is", name_dummy_normalizer)
        print()

        # The Sum Normalizer phase
        sum_normalizer = SumNormalizer()
        sum_normalizer.fit(points)
        sum_list = sum_normalizer.transform(points)
        m2.train(sum_list)
        name_sum_normalizer = cv.run_cv(sum_list, 2, m2, accuracy_score, False, True)
        print("Accuracy of SumNormalizer is", name_sum_normalizer)
        print()

        # The Min-Max Normalizer phase
        minmax_normalizer = MinMaxNormalizer()
        minmax_normalizer.fit(points)
        min_max_list = minmax_normalizer.transform(points)
        m2.train(min_max_list)
        name_min_max_normalizer = cv.run_cv(min_max_list, 2, m2, accuracy_score, False, True)
        print("Accuracy of MinMaxNormalizer is", name_min_max_normalizer)
        print()

        # The Z-Normalizer phase
        z_normalizer = ZNormalizer()
        z_normalizer.fit(points)
        z_normalizer_list = z_normalizer.transform(points)
        m2.train(z_normalizer_list)
        name_z_normalizer = cv.run_cv(z_normalizer_list, 2, m2, accuracy_score, False, True)
        print("Accuracy of ZNormalizer is", name_z_normalizer)
        if k_num == 5:
            print()


if __name__ == '__main__':
    loaded_points = load_data()
    run_knn(loaded_points)
