#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:22:54 2019

@author: vassili
"""

from extract_features import construct_data_matrices

# Note that we can try to do this with the Pipeline Makeover (see SKLearn tutorial colab)

#For now, can modify the tfidf, etc. stuff in extract_features, but really should do pipeline 
#together with the fitting. 
X_train_mat, X_val_mat = construct_data_matrices()