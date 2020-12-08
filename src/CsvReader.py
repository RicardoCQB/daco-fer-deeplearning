import sys
import numpy as np
import random
import csv



class CsvReader:

    def __init__(self, images_csv_filename, has_header=True):
        self.images_csv_filename = images_csv_filename
        self.has_header = has_header
        [self.header, self.csv_lines] = self.__read_csv_file

    '''This method reads the csv file of the DatasetFolderManager and stores its lines in a list'''
    def __read_csv_file(self):
        with open(self.images_csv_filename, 'r', newline='') as csv_file:
            csv_data = list(csv.reader(csv_file))

        if self.has_header:
            header = csv_data[0]
            csv_lines = csv_data[1:]
        else:
            header = None
            csv_lines = csv_data[0:]

        return header, csv_lines

