#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:30:34 2018

@author: mohsin
"""
from __future__ import print_function

import os
import re
import gc
import itertools
from collections import Counter
import logging

import numpy as np
np.random.seed(786)  # for reproducibility
import pandas as pd

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, QuantileTransformer, MinMaxScaler, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn import metrics



import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize

import dask.dataframe as dd
from dask.multiprocessing import get

from tqdm import tqdm
tqdm.pandas(tqdm)

from TargetEncoder import TargetEncoder
from utils import FeatureSelector
from LabelEncodeWithThreshold import LabelEncodeWithThreshold
from TargetEncoderWithThresh import TargetEncoderWithThresh

    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#%%
if __name__ == "__main__":
    
    #########################################################
    ##  Set Parameters for generating categorical features ##
    #########################################################
    
    LOGGER_FILE = "prepCountFeatures.log"
    CAT_COLS = ['user_id', 
                'region', 
                'city', 
                'parent_category_name',
                'category_name', 
                'param_1', 
                'param_2', 
                'param_3', 
                'image_top_1',
                'user_type',
                'activation_date']
    
    COMB_COLS = [('user_id', 'region'),
                 ('user_id', 'city'),
                 ('user_id', 'category_name'),
                 ('user_id', 'image_top_1'),
                 ('user_id', 'activation_date'),
                 ('region', 'parent_category_name'),
                 ('region', 'category_name'),
                 ('region', 'param_1'),
                 ('region', 'param_2'),
                 ('region', 'param_3'),
                 ('region', 'image_top_1'),
                 ('region', 'user_type'),
                 ('region', 'activation_date'),
                 ('city', 'parent_category_name'),
                 ('city', 'category_name'),
                 ('city', 'param_1'),
                 ('city', 'param_2'),
                 ('city', 'param_3'),
                 ('city', 'image_top_1'),
                 ('city', 'user_type'),
                 ('city', 'activation_date'),
                 ('parent_category_name', 'param_1'),
                 ('parent_category_name', 'param_2'),
                 ('parent_category_name', 'param_3'),
                 ('parent_category_name', 'image_top_1'),
                 ('parent_category_name', 'user_type'),
                 ('parent_category_name', 'activation_date'),
                 ('category_name', 'param_1'),
                 ('category_name', 'param_2'),
                 ('category_name', 'param_3'),
                 ('category_name', 'image_top_1'),
                 ('category_name', 'user_type'),
                 ('category_name', 'activation_date'),
                 ('param_1', 'param_2'),
                 ('param_1', 'param_3'),
                 ('param_1', 'image_top_1'),
                 ('param_1', 'user_type'),
                 ('param_1', 'activation_date'),
                 ('param_2', 'image_top_1'),
                 ('param_2', 'user_type'),
                 ('param_2', 'activation_date'),
                 ('param_3', 'image_top_1'),
                 ('param_3', 'user_type'),
                 ('image_top_1', 'user_type'),
                 ('image_top_1', 'activation_date'),
                 ('user_type', 'activation_date'),
                 ('region', 'category_name', 'param_1'),
                 ('region', 'category_name', 'param_2'),
                 ('region', 'category_name', 'user_type'),
                 ('region', 'category_name', 'activation_date'),
                 ('region', 'param_1', 'param_2'),
                 ('region', 'param_1', 'user_type'),
                 ('region', 'param_1', 'activation_date'),
                 ('region', 'param_2', 'user_type'),
                 ('region', 'param_2', 'activation_date'),
                 ('region', 'user_type', 'activation_date'),
                 ('city', 'category_name', 'param_1'),
                 ('city', 'category_name', 'param_2'),
                 ('city', 'category_name', 'user_type'),
                 ('city', 'category_name', 'activation_date'),
                 ('city', 'param_1', 'param_2'),
                 ('city', 'param_1', 'user_type'),
                 ('city', 'param_1', 'activation_date'),
                 ('city', 'param_2', 'user_type'),
                 ('city', 'param_2', 'activation_date'),
                 ('city', 'user_type', 'activation_date'),
                 ('category_name', 'param_1', 'param_2'),
                 ('category_name', 'param_1', 'user_type'),
                 ('category_name', 'param_1', 'activation_date'),
                 ('category_name', 'param_2', 'user_type'),
                 ('category_name', 'param_2', 'activation_date'),
                 ('category_name', 'user_type', 'activation_date'),
                 ('param_1', 'param_2', 'user_type'),
                 ('param_1', 'param_2', 'activation_date'),
                 ('param_1', 'user_type', 'activation_date'),
                 ('param_2', 'user_type', 'activation_date'),
                 ('region', 'category_name', 'param_1', 'param_2'),
                 ('region', 'category_name', 'param_1', 'user_type'),
                 ('region', 'category_name', 'param_1', 'activation_date'),
                 ('region', 'category_name', 'param_2', 'user_type'),
                 ('region', 'category_name', 'param_2', 'activation_date'),
                 ('region', 'category_name', 'user_type', 'activation_date'),
                 ('region', 'param_1', 'param_2', 'user_type'),
                 ('region', 'param_1', 'param_2', 'activation_date'),
                 ('region', 'param_1', 'user_type', 'activation_date'),
                 ('region', 'param_2', 'user_type', 'activation_date'),
                 ('city', 'category_name', 'param_1', 'param_2'),
                 ('city', 'category_name', 'param_1', 'user_type'),
                 ('city', 'category_name', 'param_1', 'activation_date'),
                 ('city', 'category_name', 'param_2', 'user_type'),
                 ('city', 'category_name', 'param_2', 'activation_date'),
                 ('city', 'category_name', 'user_type', 'activation_date'),
                 ('city', 'param_1', 'param_2', 'user_type'),
                 ('city', 'param_1', 'param_2', 'activation_date'),
                 ('city', 'param_1', 'user_type', 'activation_date'),
                 ('city', 'param_2', 'user_type', 'activation_date'),
                 ('category_name', 'param_1', 'param_2', 'user_type'),
                 ('category_name', 'param_1', 'param_2', 'activation_date'),
                 ('category_name', 'param_1', 'user_type', 'activation_date'),
                 ('category_name', 'param_2', 'user_type', 'activation_date'),
                 ('param_1', 'param_2', 'user_type', 'activation_date')]
    

    handler = logging.FileHandler(LOGGER_FILE)
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    
    #########################################################
    ##  Read data                                          ##
    #########################################################
    logger.info("Reading data")
    train = pd.read_csv("../input/train.csv", parse_dates=['activation_date'], nrows=10000)
    test = pd.read_csv("../input/test.csv", parse_dates=['activation_date'], nrows=10000)
    test['deal_probability'] = -1

    train_active = pd.read_csv("../input/train.csv", parse_dates=['activation_date'], nrows=100000)
    test_active = pd.read_csv("../input/test.csv", parse_dates=['activation_date'], nrows=100000)
    train_active['deal_probability'] = -1 
    test_active['deal_probability'] = -1
    
    #City correction
    for df in train, test, train_active, test_active:
        df['city'] = df['region'].astype(str) + "_" + df["city"].astype(str)
        df = df.fillna(-1)
        
    y = train['deal_probability'].values
    cvlist = list(KFold(10, random_state=123).split(y))
    
    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    
    
    #########################################################
    ##  Count encode base features                         ##
    #########################################################
    logger.info("Generating count encoding features for base features")
    for col in CAT_COLS:        
        trenc = TargetEncoder(cols = [col], targetcol= 'item_id',
                                        func = 'count')
        try:
            cols = [col] + ['item_id']
            trenc.fit(pd.concat([train[cols], test[cols], train_active[cols], test_active[cols]]))
            X_train = trenc.transform(train)
            X_test = trenc.transform(test)
            
            logger.info("Saving count features for {}".format(col))
            np.save("../utility/X_train_{}_counts.npy".format(col), X_train)
            np.save("../utility/X_test_{}_counts.npy".format(col), X_test)
        except:
            logger.info("Could not find a valid transformation")
            continue
        
    #########################################################
    ##  Count encode combination features                  ##
    #########################################################
    logger.info("Generating count encoding features for base features")
    for cols in COMB_COLS:
        trenc = TargetEncoder(cols = cols, targetcol= 'item_id',
                                        func = 'count')
        try:
            acols = cols + ['item_id']
            trenc.fit(pd.concat([train[acols], test[acols], train_active[acols], test_active[acols]]))
            X_train = trenc.transform(train)
            X_test = trenc.transform(test)
            
            logger.info("Saving count features for {}".format(col))
            np.save("../utility/X_train_{}_counts.npy".format(col), X_train)
            np.save("../utility/X_test_{}_counts.npy".format(col), X_test)
        except:
            logger.info("Could not find a valid transformation")
            continue
     
    handler.close()
    logger.removeHandler(handler)