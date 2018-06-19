#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 16:43:54 2018

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
from utils import *
from LabelEncodeWithThreshold import LabelEncodeWithThreshold
from TargetEncoderWithThresh import TargetEncoderWithThresh

    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create a file handler


#%%
if __name__ == "__main__":
    
    #########################################################
    ##  Set Parameters for generating categorical features ##
    #########################################################
    
    LOGGER_FILE = "prepCategoricalFeatures.log"
    CAT_COLS = ['user_id', 
                'region', 
                'city', 
                'parent_category_name',
                'category_name', 
                'param_1', 
                'param_2', 
                'param_3', 
                'image_top_1',
                'user_type']
    
    COMB_COLS = [
                 ('region', 'parent_category_name'),
                 ('region', 'category_name'),
                 ('region', 'param_1'),
                 ('region', 'param_2'),
                 ('region', 'param_3'),
                 ('region', 'image_top_1'),
                 ('region', 'user_type'),
                 ('city', 'parent_category_name'),
                 ('city', 'category_name'),
                 ('city', 'param_1'),
                 ('city', 'param_2'),
                 ('city', 'param_3'),
                 ('city', 'image_top_1'),
                 ('city', 'user_type'),
                 ('parent_category_name', 'param_1'),
                 ('parent_category_name', 'param_2'),
                 ('parent_category_name', 'param_3'),
                 ('parent_category_name', 'image_top_1'),
                 ('parent_category_name', 'user_type'),
                 ('category_name', 'param_1'),
                 ('category_name', 'param_2'),
                 ('category_name', 'param_3'),
                 ('category_name', 'image_top_1'),
                 ('category_name', 'user_type'),
                 ('param_1', 'param_2'),
                 ('param_1', 'param_3'),
                 ('param_1', 'image_top_1'),
                 ('param_1', 'user_type'),
                 ('param_2', 'param_3'),
                 ('param_2', 'image_top_1'),
                 ('param_3', 'image_top_1'),
                 ('image_top_1', 'user_type'),
                 ('region', 'category_name', 'param_1'),
                 ('user_type', 'region', 'parent_category_name'),
                 ('user_type', 'region', 'category_name', 'param_1'),
                 ('user_type', 'region', 'category_name', 'param_1', 'param_2'),
                 ('user_type', 'city', 'category_name', 'param_1'),
                 ('user_type', 'city', 'category_name', 'param_1', 'param_2'),]
    
    BASE_ENC_THRESH = [1,2,5,8]
    COMB_ENC_THRESH = [3, 8]
    TARGET_ENC_BASE_THRESH = [1, 3]
    TARGET_ENC_COMB_THRESH = [3, 8]

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
    train = pd.read_csv("../input/train.csv", parse_dates=['activation_date'])
    test = pd.read_csv("../input/test.csv", parse_dates=['activation_date'])
    test['deal_probability'] = -1
    
    train["image_top_1"] = train["image_top_1"].astype(str)
    test["image_top_1"] = test["image_top_1"].astype(str)    
    #City correction
    for df in train, test:
        df['city'] = df['region'].astype(str) + "_" + df["city"].astype(str)
        df = df.fillna(-1)
        
    y = train['deal_probability'].values
    cvlist = list(KFold(5, random_state=123).split(y))
    
    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    
    
    #########################################################
    ##  Label Encode features                              ##
    #########################################################
    logger.info("Generating base label encoding features")
    for (col, thresh) in list(itertools.product(CAT_COLS, BASE_ENC_THRESH)):
        print(col, thresh)
        fs = FeatureSelector(col)
        lbenc = LabelEncodeWithThreshold(thresh = thresh, logger = logger)
        pipe = make_pipeline(fs, lbenc)
        try:
            X_train = pipe.fit_transform(train)
            X_test = pipe.transform(test)
            
            logger.info("Saving label encoded features for {}".format(col))
            np.save("../utility/X_train_{}_lbenc_{}.npy".format(col, thresh), X_train)
            np.save("../utility/X_test_{}_lbenc_{}.npy".format(col, thresh), X_test)
        except:
            logger.info("Could not find a valid transformation")
            continue
     
        
    #########################################################
    ##  Label Encode categorical features                  ##
    #########################################################
    logger.info("Generating combination label encoding features")
    for cols, thresh in list(itertools.product(COMB_COLS, COMB_ENC_THRESH)):
        print(cols, thresh)
        col = "_".join(cols)
        for df in train, test:
            df[col] = df[list(cols)].apply(lambda x: "_".join([str(cc) for cc in x ]), axis=1)
            
        fs = FeatureSelector(col)
        lbenc = LabelEncodeWithThreshold(thresh = thresh, logger = logger)
        pipe = make_pipeline(fs, lbenc)
        try:
            X_train = pipe.fit_transform(train)
            X_test = pipe.transform(test)
            
            logger.info("Saving label encoded features for {} and thresh {}".format(col, thresh))
            np.save("../utility/X_train_{}_lbenc_{}.npy".format(col, thresh), X_train)
            np.save("../utility/X_test_{}_lbenc_{}.npy".format(col, thresh), X_test)
        except:
            logger.info("Could not find a valid transformation")
            continue    
    
    
    #########################################################
    ##  Target Mean Encode categorical features            ##
    #########################################################
    logger.info("Generating target encoding features")
    #train = train.fillna(-1)
    #test = test.fillna(-1)
    for (col, thresh) in list(itertools.product(CAT_COLS, TARGET_ENC_BASE_THRESH)):
        #print(col, thresh)
        if col != "user_id":
            trenc = TargetEncoderWithThresh(cols = [col], targetcol= 'deal_probability',
                                            thresh = thresh, func = 'mean')
            try:
                X_train = cross_val_predict(trenc, train, y, cv = cvlist, method = 'transform', n_jobs=1)
                X_test = trenc.fit(train).transform(test)
                
                logger.info("Saving target encoded features for {}, thresh: {}".format(col, thresh))
                np.save("../utility/X_train_{}_trenc_{}.npy".format(col, thresh), X_train)
                np.save("../utility/X_test_{}_trenc_{}.npy".format(col, thresh), X_test)
            except:
                logger.info("Could not find a valid transformation")
                continue 
        else:
            tmp = train.sort_values(by="activation_date")
            X_train = tmp.groupby("user_id")["deal_probability"].apply(lambda x: x.expanding().mean().shift().fillna(-1)).sort_index()
            
            trenc = TargetEncoderWithThresh(cols = [col], targetcol= 'deal_probability',
                                            thresh = 1, func = 'mean')
            X_test = trenc.fit(train).transform(test)
            X_test[np.isnan(X_test)] = -1
            logger.info("Saving target encoded features for {}, thresh: {}".format(col, thresh))
            np.save("../utility/X_train_{}_trenc_{}.npy".format(col, thresh), X_train)
            np.save("../utility/X_test_{}_trenc_{}.npy".format(col, thresh), X_test)
            del tmp
            #break
    #########################################################
    ##  Target Mean Encode categorical combination features##
    #########################################################
    logger.info("Generating target encoding features for combinations")
    for cols, thresh in list(itertools.product(COMB_COLS, TARGET_ENC_COMB_THRESH)):
        col = "_".join(cols)          
        trenc = TargetEncoderWithThresh(cols = list(cols), targetcol= 'deal_probability',
                                        thresh = thresh, func = 'mean')
        try:
            X_train = cross_val_predict(trenc, train, y, cv = cvlist, method = 'transform', n_jobs=1)
            X_test = trenc.fit(train).transform(test)
            
            logger.info("Saving label encoded features for {} and thresh {}".format(col, thresh))
            np.save("../utility/X_train_{}_trenc_{}.npy".format(col, thresh), X_train)
            np.save("../utility/X_test_{}_trenc_{}.npy".format(col, thresh), X_test)
        except:
            logger.info("Could not find a valid transformation")
            continue

   
    #########################################################
    ##  Target std Encode categorical features##
    #########################################################
    logger.info("Generating target encoding features")
    for (col, thresh) in list(itertools.product(CAT_COLS, TARGET_ENC_BASE_THRESH)):
        
        if col == "user_id":
            continue
        trenc = TargetEncoderWithThresh(cols = [col], targetcol= 'deal_probability',
                                        thresh = thresh, func = 'std')
        try:
            X_train = cross_val_predict(trenc, train, y, cv = cvlist, method = 'transform', n_jobs=1)
            X_test = trenc.fit(train).transform(test)
            
            logger.info("Saving target encoded features for {}, thresh: {}".format(col, thresh))
            np.save("../utility/X_train_{}_trencSD_{}.npy".format(col, thresh), X_train)
            np.save("../utility/X_test_{}_trencSD_{}.npy".format(col, thresh), X_test)
        except:
            logger.info("Could not find a valid transformation")
            continue  
        
        
    handler.close()
    logger.removeHandler(handler)
