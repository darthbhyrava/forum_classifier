import numpy as np
import re
import itertools
from collections import Counter
import csv

train_file = "./dataset/ICHI2016-TrainData.tsv"
test_file = "./dataset/new_ICHI2016-TestData_label.tsv"

def load_data_and_labels(data_file=train_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    """
    There are 7 categories - 
    1. DEMO
    2. DISE
    3. TRMT
    4. GOAL
    5. PREG
    6. FMLY
    7. SOCL
    """
    d = {}
    d['DEMO'] = [1, 0, 0, 0, 0, 0, 0]
    d['DISE'] = [0, 1, 0, 0, 0, 0, 0]
    d['TRMT'] = [0, 0, 1, 0, 0, 0, 0]
    d['GOAL'] = [0, 0, 0, 1, 0, 0, 0]
    d['PREG'] = [0, 0, 0, 0, 1, 0, 0]
    d['FAML'] = [0, 0, 0, 0, 0, 1, 0]
    d['SOCL'] = [0, 0, 0, 0, 0, 0, 1]

    max_len = -1

    #Load data from files
    samples = []
    with open(data_file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for i, row in enumerate(spamreader):
            if (row[0] == "Category"):
                continue
            print (i, row[1])
            #samples.append([row[0], row[2]])
            #getting class and title = row[0] and row[1] respectively
            samples.append([row[1], row[2], row[0]])
    #split by words

    return samples

if __name__ == "__main__":

    s = load_data_and_labels()
    s2 = load_data_and_labels(test_file)
    print("Done")
