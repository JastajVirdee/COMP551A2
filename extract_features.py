#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:40:48 2019

@author: vassili
"""
import time
import os
import nltk, re, pprint
from nltk import word_tokenize
import numpy as np
import pandas as pd
#np.set_printoptions(threshold=np.inf)

#/home/vassili/Desktop/COMP551A2.git/data

#this is all jerryrigged starting from the nltk documentation at
#https://www.nltk.org/book/
#chapters 1 and 6 mostly 

#this is straight up copy pasted and built on

#file_list=os.listdir('data/train/pos')
porter = nltk.PorterStemmer()
num_top_words=2000;
num_files_to_read_per_sent=100;


def load_raw_data():
    documents=[]
    all_words=[]
    for foldername in ['pos', 'neg']:
        file_list=os.listdir('data/train/'+ foldername)
        for fname in file_list[0:num_files_to_read_per_sent]:
            path='data/train/'+foldername+'/'+fname
            #f=open('data/train/pos/0_9.txt')
            f=open(path)
            raw=f.read()
            tokens=word_tokenize(raw)
            tokens=[porter.stem(t) for t in tokens]
            all_words=all_words+tokens
            documents.append((tokens, foldername))
    return documents, all_words

documents, all_words=load_raw_data()



all_words = nltk.FreqDist(w.lower() for w in all_words)
word_features = list(all_words)[:num_top_words]



num_features=3
def document_features(document): 
    """Extract document features. 
    Input: document: list of lists: first list, words in document (tokenized), second the pos/neg
    Assumes global variables: all_words, word_features
    Returns features (which are the predictors) and the corresponding outputs"""
    
    document_words = set(document[0])
    word_abs_freq=nltk.FreqDist(w.lower() for w in document[0])
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = [(word in document_words)] #binary occurence
        if features['contains({})'.format(word)]:
            features['abs_freq({})'.format(word)] = [word_abs_freq[word]] #absolute frequency
            features['rel_freq({})'.format(word)] = [word_abs_freq[word]/len(document[0])] #relative frequency
        else:
            features['abs_freq({})'.format(word)] = [0] #absolute frequency
            features['rel_freq({})'.format(word)] = [0] #absolute frequency
    return features, document[1]

start = time.time()
feat_list=[]
review_list=[]
features=pd.DataFrame()
reviews=pd.DataFrame()

buffer_size=200;

for i in range(len(documents)):

    feat, review=document_features(documents[i])
    feat_list.append(feat)
    review_list.append(review_list)
    
    if i % buffer_size == 0 or i==(len(documents)-1):
        end=time.time()
        print("Example {}, current time taken: {}".format(i, end-start))
        feat_df=pd.DataFrame.from_dict(feat_list)
        review_df=pd.DataFrame.from_dict(review_list)
        features=features.append(feat_df)   
        reviews=reviews.append(review_df)
        feat_list=[]
        review_list=[]




        

end=time.time()
print("Time taken:{}".format(end-start))


#for word in word_features:
 #   print(word)
  #  print(word in document_words)
    




























































