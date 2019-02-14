#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:22:54 2019

@author: vassili
"""

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



"""Get back data matrices for the pos/neg examples. """
corpus, reviews=load_raw_data(num_files_to_read_per_sent=4000) #HOW MANY FILES TO LOAD

X_train, X_val, y_train, y_val = train_test_split(corpus, reviews, train_size=0.8, test_size=0.2)

#This can take the place of CountVectorizer below - it lemmatizes.
#comparisons could be interesting
lemma_vect = CountVectorizer(tokenizer=MyTokenizer()) #Use a lemmatizer, add to the countvectorizer


#This is a problem..Run this, bottom for loop detects two distinct class labels, here only one. Why?
pclf = Pipeline([
    ('vect', lemma_vect),
    ('tfidf', TfidfTransformer()),
    ('norm', Normalizer()),
    ('clf', MultinomialNB()),
])

pclf.fit(X_train, y_train)
y_pred = pclf.predict(X_val)

print(metrics.classification_report(y_val, y_pred,
    target_names=["Review"]))

#Compare classifiers on this dataset with this lemmatization, 
#Very similar to what is done at the scikit learn link below
#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
names=["MultinomialNB", "SVM"]
classifiers=[
        MultinomialNB(),
        SVC()
        ]

rng = np.random.RandomState(2)

for name, clf in zip(names, classifiers):
    pclf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('norm', Normalizer()),
    ('clf', clf),
    ])
    pclf.fit(X_train, y_train)
    y_pred = pclf.predict(X_val)
    score = pclf.score(X_val, y_val)
    print("Model results for {}".format(name))
    print("Accuracy:{}".format(score))
    print(metrics.classification_report(y_val, y_pred,
    target_names=["Positive", "Negative"])) # not sure about this, these should be the class labels































