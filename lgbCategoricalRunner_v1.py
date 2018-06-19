#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 13:32:23 2018

@author: mohsin
"""

import gc
import os
from copy import copy 

import pandas as pd
import numpy as np
np.random.seed(786)

from utils import *
from sklearn.model_selection import KFold, cross_val_predict
import logging

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm
tqdm.pandas(tqdm)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#%%
def cv_oof_predictions(estimator, X, y, cvlist, est_kwargs, fit_params, predict_test=True, X_test=None, ):
    preds = np.zeros(len(y)) #Initialize empty array to hold prediction
    test_preds = []
    for tr_index, val_index in cvlist:
        gc.collect()
        X_tr , X_val = X[tr_index], X[val_index]
        y_tr , y_val = y[tr_index], y[val_index]
        est = estimator.set_params(**est_kwargs)
        #print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)
        est.fit(X_tr, y_tr, eval_set = [(X_tr, y_tr), (X_val, y_val)], eval_metric='rmse',
               early_stopping_rounds=50, verbose=200, **fit_params) #Might need to change this depending on estimator
        preds[val_index] = est.predict(X_val)
        #break
        if predict_test:
            tpreds = est.predict(X_test)
            test_preds.append(tpreds)
        
    if len(test_preds) >0:
        test_preds = np.mean(test_preds, axis=0)
    return est, preds, test_preds

    
#%%
if __name__ == "__main__":
    
    ############ Models to try ################################################
    LOGGER_FILE = "lgbCategoricalRunner_v1.log"
    CONT_COLS = ['price', 'item_seq_number', 'user_id_counts', 'price_binned']
    
    BASE_FEATURES = ['region_lbenc_2', 'city_lbenc_2', 'parent_category_name_lbenc_2', 
                 'category_name_lbenc_2', 'param_1_lbenc_2', 'param_2_lbenc_2', 'param_3_lbenc_2', 
                 'image_top_1_lbenc_2', 'user_type_lbenc_2', 'user_id_trenc_3', 'city_trenc_3',
                 'parent_category_name_trenc_3', 'param_1_trenc_3', 'param_2_trenc_3', 'param_3_trenc_3',
                 'image_top_1_trenc_3', 'region_parent_category_name_trenc_8', 'region_param_2_trenc_8',
                 'region_param_3_trenc_8', 'region_image_top_1_trenc_8', 'city_parent_category_name_trenc_8',
                 'city_category_name_trenc_8', 'city_param_1_trenc_8', 'city_param_3_trenc_8', 
                 'parent_category_name_param_2_trenc_8', 'parent_category_name_param_3_trenc_8',
                 'parent_category_name_image_top_1_trenc_8', 'parent_category_name_user_type_trenc_8',
                 'category_name_param_1_trenc_8', 'category_name_image_top_1_trenc_8', 
                 'param_1_image_top_1_trenc_8', 'param_2_image_top_1_trenc_8', 
                 'user_type_region_category_name_param_1_trenc_8', 
                 'user_type_city_category_name_param_1_trenc_8']
    
    PRICE_COMB_COLS = [
                 'price_binned_param_2_trenc_5'
                 ]
    
    PRICE_MEAN_COLS = [
            'category_name_priceenc_5', 'param_1_priceenc_5', 'param_2_priceenc_5', 
            'param_3_priceenc_5', 'image_top_1_priceenc_5'
            ]
    
    COUNT_COLS = ['param_1_counts', 'user_type_counts', 'param_1_user_type_activation_date_counts',
                  'param_2_user_type_activation_date_counts',
                  'region_param_1_user_type_activation_date_counts', 
                  'city_category_name_param_1_user_type_counts', 
                  'city_category_name_param_2_user_type_counts']
    

    LGB_PARAMS1 = {
            "n_estimators":10000,
            'learning_rate': 0.02,
            "num_leaves":255,
            "colsample_bytree": 0.4,
            "subsample": 0.9,
            "reg_alpha": 0,
            "reg_lambda": 0,
            "min_data_in_leaf": 200,
            "max_bin": 255,
            "verbose":0
            }
    
    
    LGB_PARAMS2 = {
            "n_estimators":10000,
            'learning_rate': 0.01,
            "num_leaves":300,
            "colsample_bytree": 0.5,
            "subsample": 0.8,
            "reg_alpha": 1,
            "reg_lambda": 5,
            "min_data_in_leaf": 50,
            "max_bin": 512,
            "verbose":0
            }
    
    LGB_PARAMS2 = {
            "n_estimators":10000,
            'learning_rate': 0.02,
            "num_leaves":127,
            "colsample_bytree": 0.7,
            "subsample": 0.9,
            "reg_alpha": 0,
            "reg_lambda": 0,
            "min_data_in_leaf": 500,
            "max_bin": 255,
            "verbose":0
            }
    LGB_PARAMS = [LGB_PARAMS1, LGB_PARAMS2]
    
    
    ######################   Logger   #########################################
    handler = logging.FileHandler(LOGGER_FILE)
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    ###################### Read data ##########################################
    logger.info("Reading data")
    train = pd.read_csv("../input/train.csv", parse_dates=['activation_date'])
    test = pd.read_csv("../input/test.csv", parse_dates=['activation_date'])
    test['deal_probability'] = -1
    
    #City correction
    for df in train, test:
        df['city'] = df['region'].astype(str) + "_" + df["city"].astype(str)
        df = df.fillna(-1)
        
    y = train['deal_probability'].values
    cvlist = list(KFold(5, random_state=123).split(y))
    
    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    del train, test
    
    ################### Greedy forward feature selection ######################
    #
                       
    features = BASE_FEATURES + CONT_COLS + PRICE_COMB_COLS + PRICE_MEAN_COLS + COUNT_COLS
    X = np.vstack([np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r') for col in features]).T
    X_test = np.vstack([np.load("../utility/X_test_{}.npy".format(col), mmap_mode='r') for col in features]).T
    logger.info("Shape for train and test dataset is {} and {}".format(X.shape, X_test.shape))
    
    model = lgb.LGBMRegressor(n_estimators=10000)
    for i, lgb_params in enumerate(LGB_PARAMS):
        est, y_preds, y_test = cv_oof_predictions(model, X, y, cvlist, predict_test=True, 
                                           X_test=X_test, est_kwargs=lgb_params, fit_params={})
        score = rmse(y, y_preds)
        logger.info("Score for {}th lgb parameter set is {}".format(i, score))
        
        np.save("../outputs/X_train_lgbcats_{}.npy".format(i), y_preds)
        np.save("../outputs/X_test_lgbcats_{}.npy".format(i), y_test)
    
    handler.close()
    logger.removeHandler(handler)
    
    
    
    
    
    
    