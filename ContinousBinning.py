#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:04:31 2018

@author: mohsin
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ContinousBinning(BaseEstimator, TransformerMixin):
    def __init__(self, bins= 20, start = None, end = None, bin_array = None):
        self.bins= bins
        self.start = start
        self.end = end
        self.bin_array = bin_array
        
    def fit(self, X, y=None):
        if not(isinstance(X, pd.Series)):
            raise TypeError("Need pandas series as input")

        if not(self.start):
            self.start = np.min(X)
        
        if not(self.end):
            self.end = np.max(X)
        
        if not(self.bin_array):
            self.bin_array = np.linspace(self.start, self.end, self.bins)

        return self
    
    def transform(self, X, y=None):
        if not(isinstance(X, pd.Series)):
            raise TypeError("Need pandas series as input")
            
        X_transformed = pd.cut(X, self.bin_array)
        return X_transformed
