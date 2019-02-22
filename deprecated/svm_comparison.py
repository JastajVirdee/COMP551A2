#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:48:47 2019

@author: vassili
"""

from sklearn.model_selection import GridSearchCV
from extract_features import load_raw_data
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


import numpy as np
from sklearn.svm import SVC, LinearSVC
from matplotlib import pyplot as plt

corpus, reviews=load_raw_data(num_files_to_read_per_sent=12500) #HOW MANY FILES TO LOAD

vectorizer=TfidfVectorizer()
c_vect = vectorizer.fit(corpus)
X=c_vect.transform(corpus)
Y=reviews;


param_grid = {'clf__C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'clf__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

C_arr=np.linspace(0.1, 4, 50)

param_grid = {'clf__C': C_arr.tolist()}


pclf = Pipeline([ #create sequence of transforms and classifier
    ('norm', Normalizer()),
    ('clf', LinearSVC())
])



pclf=GridSearchCV(pclf, param_grid, cv=4, verbose=1)
pclf.fit(X, Y)
max_idx=np.argmax(pclf.cv_results_['mean_test_score'])
plt.figure()
plt.plot(C_arr, pclf.cv_results_['mean_test_score'], label="Accuracy")
plt.scatter(C_arr[max_idx], pclf.cv_results_['mean_test_score'][max_idx],color='r',
            label="C={}".format(C_arr[max_idx]))
plt.xlabel("C value", fontsize=13)
plt.ylabel("Cross Validation Accuracy", fontsize=13)
plt.savefig("LinearSVM_C_Comparison.pdf",bbox_inches='tight')
plt.legend()







