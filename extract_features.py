#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:03:22 2019

@author: vassili
"""

import os
import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



class MyTokenizer(object):
    def __init__(self, Stemmer):
        """Give this either 
        WordNetLemmatizer()
        or nltk.PorterStemmer()"""
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
    

tokenizer=MyTokenizer(WordNetLemmatizer())
lemma_vect = CountVectorizer(tokenizer=MyTokenizer(WordNetLemmatizer()))

num_files_to_read_per_sent=4000;

def load_raw_data():
    corpus=[]
    reviews=[]
    for foldername in ['pos', 'neg']:
        file_list=os.listdir('data/train/'+ foldername)
        for fname in file_list[0:num_files_to_read_per_sent]:
            path='data/train/'+foldername+'/'+fname
            #f=open('data/train/pos/0_9.txt')
            f=open(path)
            raw=f.read()
            corpus.append(raw)
            reviews.append(1 if foldername=='pos' else 0)
    return corpus, reviews

corpus, reviews=load_raw_data()

X_train, X_val, y_train, y_val = train_test_split(corpus, reviews, train_size=0.8, test_size=0.2)


count_vect = lemma_vect.fit(X_train)
X_train_counts = count_vect.transform(X_train)
X_val_counts = count_vect.transform(X_val) 

tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_val_tfidf = tfidf_transformer.transform(X_val_counts)


X_train=hstack([X_train_counts, X_train_tfidf]).toarray()
X_val=hstack([X_val_counts, X_val_tfidf]).toarray()



#    tokens=word_tokenize(raw)
#tokens=[porter.stem(t) for t in tokens]

#tokens=word_tokenize(raw)
#tokens=[porter.stem(t) for t in tokens]
#all_words=all_words+tokens
#documents.append((tokens, foldername))










