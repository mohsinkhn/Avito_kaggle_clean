#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 08:36:31 2018

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
from tqdm import tqdm
tqdm.pandas(tqdm)

from utils import *
from LabelEncodeWithThreshold import LabelEncodeWithThreshold
from TargetEncoderWithThresh import TargetEncoderWithThresh
from ContinousBinning import ContinousBinning
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#%%
if __name__ == "__main__":
    
    #########################################################
    ##  Set Parameters for generating categorical features ##
    #########################################################
    
    LOGGER_FILE = "prepCcontinousFeatures.log"
    CONT_COLS = ["price", "item_seq_number"]
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
                 ('price_binned', 'param_1_2_3'),
                 ('price_binned', 'param_2'),
                 ('price_binned', 'category_name'),
                 ('price_binned', 'param_1'),
                 ('price_binned', 'param_3'),
                 ('price_binned', 'parent_category_name'),
                 ('price_binned', 'image_top_1'),
                 ('price_binned', 'user_type'),
                 ('price_binned', 'region', 'parent_category_name'),
                 ('price_binned', 'region', 'category_name'),
                 ('price_binned', 'city', 'category_name'),
                 ('price_binned', 'user_type', 'city', 'category_name'),
                 ('price_binned', 'user_type', 'city', 'category_name', 'param_1'),
                 ('price_binned', 'user_type', 'city', 'category_name', 'param_2')
                 ]
    
    TARGET_ENC_COMB_THRESH = [5]

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
        df["param_1_2_3"] = df["param_1"].astype(str) + "_" + df["param_2"].astype(str) + "_" + df["param_3"].astype(str) 
    
    y = train['deal_probability'].values
    cvlist = list(KFold(5, random_state=123).split(y))
    
    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    
    
    #########################################################
    ##  Clean and Bin price                                ##
    #########################################################    
    logger.info("Cleaning and binning price")
    for df in train, test:
        df["price"] = df["price"].clip(0, 10**8)
        df["isna_price"]  = df["price"].isnull()
        
    logger.info("Fill missing price with category mean")
    
    price_cat_means = train.groupby("category_name")["price"].mean()
    for df in train, test:
        df["cat_price_mean"] = df.category_name.map(price_cat_means)
        df["price"] = df["price"].fillna(df["cat_price_mean"])
        
    logger.info("Missing values for price in train and test after imputation are \
                {}  and {}".format(train["price"].isnull().sum(), test["price"].isnull().sum()))
    
    
    logger.info("")
    
    logger.info("Binning price")
    binner = ContinousBinning(40)
    lbenc  = LabelEncodeWithThreshold(thresh=1)
    pipe = make_pipeline(binner, lbenc)
    
    train["price_binned"] = pipe.fit_transform(train["price"])
    test["price_binned"] = pipe.transform(test["price"])
    
    np.save("../utility/X_train_price.npy", train["price"])
    np.save("../utility/X_test_price.npy", test["price"])
    
    np.save("../utility/X_train_price_binned.npy", train["price_binned"])
    np.save("../utility/X_test_price_binned.npy", test["price_binned"])
    #########################################################
    ##  Target encoding for price binned and categorical cols                                ##
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
    ## Item seq number features                            ##
    #########################################################
    logger.info("Processing item seq number")
    sel_cols = ["user_id", "item_seq_number"]
    
    for df in train, test:
        min_seq = df.groupby("user_id")["item_seq_number"].transform(min)
        df["user_id_counts"] = df["item_seq_number"] - min_seq
        df["item_seq_number"] = np.log1p(df["item_seq_number"].clip(2000))
        
    np.save("../utility/X_train_user_id_counts.npy", train["user_id_counts"])
    np.save("../utility/X_test_user_id_counts.npy", test["user_id_counts"])
    
    np.save("../utility/X_train_item_seq_number.npy", train["item_seq_number"])
    np.save("../utility/X_test_item_seq_number.npy", test["item_seq_number"])
    
    
    #########################################################
    ##  Median price col                               ##
    ######################################################### 
    logger.info("Generating target encoding features for combinations")
    for col, thresh in list(itertools.product(CAT_COLS, TARGET_ENC_COMB_THRESH)):
        #col = "_".join(cols)          
        trenc = TargetEncoderWithThresh(cols = [col], targetcol= 'price',
                                        thresh = thresh, func = 'mean')
        try:
            X_train = cross_val_predict(trenc, train, y, cv = cvlist, method = 'transform', n_jobs=1)
            X_test = trenc.fit(train).transform(test)
            
            logger.info("Saving label encoded features for {} and thresh {}".format(col, thresh))
            np.save("../utility/X_train_{}_priceenc_{}.npy".format(col, thresh), X_train)
            np.save("../utility/X_test_{}_priceenc_{}.npy".format(col, thresh), X_test)
        except:
            logger.info("Could not find a valid transformation")
            continue