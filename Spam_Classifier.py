from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import nltk
import csv
import os
import re

# loading the dataset

dataset = load_dataset("Deysi/spam-detection-dataset")

# Extracting training data from the datset

train_dataset_text = dataset['train']['text']
train_dataset_target = dataset['train']['label']

# Coverting the dataset targets to 1(spam) or 0(non-spam)

for i in range(0,len(train_dataset_target)):
    if train_dataset_target[i] == 'spam':
        train_dataset_target[i] = 1
    else:
        train_dataset_target[i] = 0

# Collecting all the words obtained in the emails as my dictionary

my_dict = {}

c = [0,0]
for i in train_dataset_target:
    c[i] += 1

for i in range(0,len(train_dataset_text)):
    words = nltk.word_tokenize(train_dataset_text[i])
    l = set()
    for j in words:
        if j not in l:
            l.add(j)
            y = train_dataset_target[i]
            if j in my_dict:
                my_dict[j][y] += 1
            else:
                my_dict[j] = [int(y==0),int(y==1)]

for i in my_dict:
    my_dict[i][0] = (my_dict[i][0]+1)/(c[0]+1)
    my_dict[i][1] = (my_dict[i][1]+1)/(c[1]+1)

print(len(my_dict))

# SVM Algorithm Implementation

def vec(words):
    x = []
    for i in my_dict:
        x.append(int(i in words))
    return np.array(x)

train = []
for i in range(0,len(train_dataset_target)):
    train.append(vec(nltk.word_tokenize(train_dataset_text[i])))

X = np.array(train)
y = np.array(train_dataset_target)
print(X.shape,y.shape)

svm_model = SVC(kernel='linear', C = 1.0)
svm_model.fit(X, y)
    
j = 0;test_labels = []
directory = 'test'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            content = file.read()
            words = nltk.word_tokenize(content)
            x = []
            for i in my_dict:
                x.append(int(i in words))
            predicted_label = svm_model.predict(np.array(x).reshape(1,len(x)))
            test_labels.append(predicted_label)
            if predicted_label == 0:
                print("Non-Spam")
            else:
                print("Spam")
            j += 1
            print(j)
