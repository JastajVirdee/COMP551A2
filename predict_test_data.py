#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:33:35 2019

@author: vassili
"""

import os

#TRAIN MODEL
from extract_features import MyTokenizer, load_raw_data
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt


corpus, reviews=load_raw_data(num_files_to_read_per_sent=4000) #HOW MANY FILES TO LOAD

X_train, X_val, y_train, y_val = train_test_split(corpus, reviews, train_size=0.8, test_size=0.2)

lemma_vect = CountVectorizer(tokenizer=MyTokenizer()) #Use a lemmatizer, add to the countvectorizer

myVectorizer=CountVectorizer()



clf= MultinomialNB()
pclf = Pipeline([ #create sequence of transforms and classifier
('vect', myVectorizer),
('tfidf', TfidfTransformer()),
('norm', Normalizer()),
('clf', clf),
])


pclf.fit(X_train, y_train)
y_pred = pclf.predict(X_val)



score = accuracy_score(y_val, y_pred)
print("Accuracy:{}".format(score))



def read_test_data(nfiles):
    test_corpus=[]
    for i in range(nfiles):
        path='data/test/'+'/'+str(i)+'.txt'
        f=open(path)
        raw=f.read()
        test_corpus.append(raw)
    return test_corpus
    
nfiles=25000
test_data=read_test_data(nfiles)
y_pred = pclf.predict(test_data)

idxs=np.arange(nfiles)
pred_mat=np.array([idxs, y_pred]).T

#REMEMBER TO MANUALLY ENTER HEADERS AFTER
np.savetxt("foo.csv", pred_mat, delimiter=",")



























