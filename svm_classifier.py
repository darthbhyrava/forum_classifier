from __future__ import division
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn import svm , preprocessing
from doc_helper import load_data_and_labels
import numpy as np
import pickle
import pdb

train_file = "./dataset/ICHI2016-TrainData.tsv"
test_file = "./dataset/new_ICHI2016-TestData_label.tsv"



def my_svm_classifier(train, l1, test, l2):

    print ("Training data size", train.shape)
    print ("SVM being trained")
    clf = svm.SVC()
    clf.fit(preprocessing.scale(train), l1)
    #clf.fit(train, l1)
    print ("SVM Trained successfully")
    print ("Testing data size", test.shape)

    hits = 0

    output = clf.predict(preprocessing.scale(test))
    #output = clf.predict(test))

    for i in range(len(output)):
        if l2[i] == output[i]:
            hits += 1
        print hits, "Accuracy", float(hits)/len(l2)*100
    
    print hits, "/", len(l2), "FINAL Accuracy", float(hits)/len(l2)*100

if __name__ == "__main__":
    
    d = {}
    d['DEMO'] = 1
    d['DISE'] = 2
    d['TRMT'] = 3
    d['GOAL'] = 4
    d['PREG'] = 5
    d['FAML'] = 6
    d['SOCL'] = 7

    doc_embed = pickle.load(open('./data/doc_embed.pkl', 'r'))
    
    s = load_data_and_labels()
    s2 = load_data_and_labels(test_file)

    l1 = []
    l2 = []
    train = []
    test = []

    for i, x in enumerate(s):
        Id = "question_train_"+str(i)
        train.append(doc_embed[Id])
        l1.append(d[x[2]])

    train = np.asarray(train)

    for i, x in enumerate(s2):
        Id = "question_test_"+str(i)
        test.append(doc_embed[Id])
        l2.append(d[x[2]])

    test = np.asarray(test)

    my_svm_classifier(train, l1, test, l2)

    print ("SVM code ran successfully")
