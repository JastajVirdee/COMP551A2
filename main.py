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
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt



"""Get back data matrices for the pos/neg examples. """
corpus, reviews=load_raw_data(num_files_to_read_per_sent=4000) #HOW MANY FILES TO LOAD

X_train, X_val, y_train, y_val = train_test_split(corpus, reviews, train_size=0.8, test_size=0.2)

#This can take the place of CountVectorizer below - it lemmatizes.
#comparisons could be interesting
lemma_vect = CountVectorizer(tokenizer=MyTokenizer()) #Use a lemmatizer, add to the countvectorizer

myVectorizer=lemma_vect
myVectorizer=CountVectorizer()
#Compare classifiers on this dataset with this lemmatization, 
#Very similar to what is done at the scikit learn link below
#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
names=["MultinomialNB", "SVM, lin kern", "SVM, gamma"]
classifiers=[
        MultinomialNB(),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1)
        ]
#        MultinomialNB(),
rng = np.random.RandomState(2)


"""Using TF IDF Features"""
def model_comparison_with_tfidf():
    """The following function returns the model accuracies for the models in classifiers
    with tfidf transform
    Note that this accuracy metric is not ideal. F1Score can also be used, left up to you. """
    accuracies=np.ones(len(names))
    i=0
    for name, clf in zip(names, classifiers): 
        """Here a 'pipeline' is created using the classifiers in the classifiers array
        Classfier is varied, all else is the same"""
        pclf = Pipeline([ #create sequence of transforms and classifier
        ('vect', myVectorizer),
        ('tfidf', TfidfTransformer()),
        ('norm', Normalizer()),
        ('clf', clf),
        ])
        pclf.fit(X_train, y_train)
        y_pred = pclf.predict(X_val)
        score = accuracy_score(y_val, y_pred) #HERE DEFINE WHAT SCORE U USE
        print("Model results for {}".format(name))
        print("Accuracy:{}".format(score))
        accuracies[i]=score
        i=i+1
    return accuracies

"""Using TF IDF Features"""
def model_comparison_with_word_count():
    """The following function returns the model accuracies for the models in classifiers
    with word counts
    I just threw out the tfidf step"""
    accuracies=np.ones(len(names))
    i=0
    for name, clf in zip(names, classifiers): 
        """Here a 'pipeline' is created using the classifiers in the classifiers array
        Classfier is varied, all else is the same"""
        pclf = Pipeline([ #create sequence of transforms and classifier
        ('vect', myVectorizer),
        ('norm', Normalizer()),
        ('clf', clf),
        ])
        pclf.fit(X_train, y_train)
        y_pred = pclf.predict(X_val)
        score = accuracy_score(y_val, y_pred) #HERE DEFINE WHAT SCORE U USE
        print("Model results for {}".format(name))
        print("Accuracy:{}".format(score))
        accuracies[i]=score
        i=i+1
    return accuracies


def tfidf_vs_word_count():
    #Plotting barcharts with two rectangles:
    #cannibalized off https://matplotlib.org/gallery/statistics/barchart_demo.html
    tfidf_accuracies=model_comparison_with_tfidf()
    wordcount_accuracies=model_comparison_with_word_count()
    
    index = np.arange(len(names))
    
    bar_width = 0.35
    
    fig = plt.figure()
    ax=plt.gca()
    tfidf_rects = ax.bar(index, tfidf_accuracies, bar_width, color='b',label='TF_IDF')
    wordcount_rects = ax.bar(index+bar_width, wordcount_accuracies, bar_width, color='r',label='Word_Count')
    
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Model')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(names)
    ax.legend()


"""Pull the trigger to compute all that stuff"""
tfidf_vs_word_count()


#Now let us home in on the decision tree. 


depths=np.linspace(10,100,10)
names=["{}".format(i) for i in depths]
classifiers=[DecisionTreeClassifier(max_depth=i) for i in depths]




def decision_tree_comparison():
    """The following function compares """
    accuracies=np.ones(len(names))
    i=0
    for i, (name, clf) in enumerate(zip(names, classifiers)): 
        """Note how I throw out the tf_idf transform"""
        pclf = Pipeline([
        ('vect', myVectorizer),
        ('norm', Normalizer()),
        ('clf', clf),
        ])
        pclf.fit(X_train, y_train)
        y_pred = pclf.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        print("Model results for {}".format(name))
        print("Accuracy:{}".format(score))
        accuracies[i]=score
        i=i+1
    return accuracies

def decision_tree_comparison_plot():
#Plotting barcharts with two rectangles:
#cannibalized off https://matplotlib.org/gallery/statistics/barchart_demo.html
    """Plot accuracy of decision tree as a function of decision tree depth"""
    accuracies=decision_tree_comparison()
    
    index = np.arange(len(names))
    
    bar_width = 0.35
    
    fig = plt.figure()
    ax=plt.gca()
    rects = ax.bar(index, accuracies, bar_width, color='b',label='Dec_tree')
    
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Model')
    ax.set_xticks(index)
    ax.set_xticklabels(names)




decision_tree_comparison_plot()






















