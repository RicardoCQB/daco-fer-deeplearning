from collections import Counter


class DatasetFER2013:

    def __init__(self, csv_lines, header):
        self.csv_lines = csv_lines
        self.header = header

    '''This method counts the number of elements of each class'''
    def count_class(self, label_name):
        label_col_index = self.header.index(label_name)
        class_counter = Counter(self.csv_lines[:, label_col_index])

        return class_counter

    '''This method dictates how many images of each class and in each fold'''
    def distribute_classes_folds(self, num_folds):
        # Maybe delete this class later.
        pass
