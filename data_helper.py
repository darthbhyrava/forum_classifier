import numpy as np
import re
import itertools
from collections import Counter
import csv

train_file = "./dataset/ICHI2016-TrainData.tsv"
test_file = "./dataset/new_ICHI2016-TestData_label.tsv"


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


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
            samples.append([row[0], row[1]])
    #split by words
    x_text = [s[1].strip() for s in samples]
    x_text = [clean_str(sent) for sent in x_text]
    sm = 0
    cnt = 0
    for x in x_text:
        sm += len(x)
        if len(x) < 35:
            cnt += 1
        max_len = max(max_len, len(x))
    print ("max length", max_len)
    print ("avg length", float(sm)/len(x_text))
    print ("cnt", cnt)
    #generate labels
    labels = [d[cat[0]] for cat in samples]
    #y = np.array(labels)

    return [x_text, labels]

'''
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
'''

if __name__ == "__main__":

    x, y = load_data_and_labels()
    print("Done")
