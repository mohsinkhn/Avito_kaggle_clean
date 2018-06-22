#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:45:42 2018

@author: mohsin
"""

import re
import string

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextStatsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer t extract text features. It extracts following information:
    - Length of text field
    - No. of words
    - No. of digits
    - No. of Cap words
    - No. of punctuation
    - No. of emoji's
    
    It also computes following ratio's if flag is True
    """
    
    def __init__(self, ratios = True, puncts= set(string.punctuation), 
                 emojis = None, logger = None):
        self.ratios = ratios
        self.puncts = puncts
        self.emojis = emojis
        self.logger = logger
        
    def __num_puncts(self, x):
        try:
            #return len(regex.findall(r"\p{P}", x, overlapped=True))
            return len([c for c in str(x) if c in self.puncts])
        except:
            return 0
        
    def __num_upper(self, x):
        try:
            #return len(regex.findall(r"\p{Lu}", x, overlapped=True))
            return len(re.findall(r"[A-ZА-Я]", x))
        except:
            return 0
    
    def __num_emoji(self, x):
        return sum([c in self.emojis for c in x])
    
    def __num_unique(self, x):
        return len(set(str(x).split(' ')))
    
    def get_feature_names(self):
        features = ["char_len", "num_words", "unq_words", "num_digits", "num_caps", "num_puncts", "num_emojis"]
        if self.ratios:
            features += ["unq_tot_word_ratio", "word_len_ratio", "digits_len_ratio", "caps_len_ratio", "punct_len_ratio"]
        return features
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.logger:
            self.logger.info("Getting text features")
        
        X = pd.Series(X).fillna(" ")
        X_len = X.str.len().astype(int)
        X_words = X.str.split(' ').str.len().astype(int)
        X_unqs  = X.apply(self.__num_unique).astype(int)
        X_digits = X.str.count("\d").astype(int)
        X_uppers = X.apply(self.__num_upper).astype(int)
        X_puncts = X.apply(self.__num_puncts).astype(int)
        X_emojis = X.apply(self.__num_emoji).astype(int)
        
        X_transformed = np.vstack((X_len, X_words, X_unqs, X_digits, X_uppers, X_puncts, X_emojis)).T
        #print(X_transformed.shape)
        if self.ratios:
            X_words_unqs = X_unqs/(X_words + 1)
            X_words_len = X_words/(X_len + 1)
            X_digits_len = X_digits/(X_len + 1)
            X_uppers_len = X_uppers/(X_len + 1)
            X_puncts_len = X_puncts/(X_len + 1)
            
            X_ratios = np.vstack((X_words_unqs, X_words_len, X_digits_len, X_uppers_len, X_puncts_len)).T
            #print(X_ratios.shape)
            X_transformed = np.hstack((X_transformed, X_ratios))
    
        return X_transformed
        