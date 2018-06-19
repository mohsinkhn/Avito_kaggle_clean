#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 22:27:01 2018

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
               early_stopping_rounds=50, verbose=0, **fit_params) #Might need to change this depending on estimator
        preds[val_index] = est.predict(X_val)
        #break
        if predict_test:
            tpreds = est.predict(X_test)
            test_preds.append(tpreds)
        
    if len(test_preds) >0:
        test_preds = np.mean(test_preds, axis=0)
    return est, preds, test_preds


def eval_lgb_sets(X, y, cvlist, param_sets):
    model = lgb.LGBMRegressor(n_estimators=10000)
    y_preds_best = np.zeros(len(X))
    best_rmse = 1.0
    best_i = 0
    for i, lgb_params in enumerate(param_sets):
        est, y_preds, _ = cv_oof_predictions(model, X, y, cvlist, predict_test=False, 
                                           X_test=None, est_kwargs=lgb_params, fit_params={})
        score = rmse(y, y_preds)
        logger.info("Score for {}th lgb parameter set is {}".format(i, score))
        if score < best_rmse:
            best_rmse = score
            y_preds_best = y_preds
            best_i = i
            
    logger.info("Best score is {} with set {}".format(best_rmse, best_i))
    return best_rmse, y_preds_best
    
#%%
if __name__ == "__main__":
    
    LOGGER_FILE = "lgbCountSelector.log"
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

    LGB_PARAMS1 = {
            "n_estimators":5000,
            'learning_rate': 0.02,
            "num_leaves":255,
            "colsample_bytree": 0.8,
            "subsample": 0.9,
            "reg_alpha": 1,
            "reg_lambda": 1,
            "min_data_in_leaf": 100,
            "max_bin": 255,
            "verbose":0
            }
    
    LGB_PARAMS2 = {
            "n_estimators":5000,
            'learning_rate': 0.02,
            "num_leaves":127,
            "colsample_bytree": 0.7,
            "subsample": 0.7,
            "reg_alpha": 0,
            "reg_lambda": 0,
            "min_data_in_leaf": 200,
            "max_bin": 512,
            "verbose":0
            }
    
    LGB_PARAMS3 = {
            "n_estimators":5000,
            'learning_rate': 0.02,
            "num_leaves":63,
            "colsample_bytree": 0.5,
            "subsample": 0.8,
            "reg_alpha": 0,
            "reg_lambda": 0,
            "min_data_in_leaf": 500,
            "max_bin": 255,
            "verbose":0
            }
    
    LGB_PARAMS4 = {
            "n_estimators":5000,
            'learning_rate': 0.02,
            "num_leaves":255,
            "colsample_bytree": 0.6,
            "subsample": 0.9,
            "reg_alpha": 1,
            "reg_lambda": 1,
            "min_data_in_leaf": 1000,
            "max_bin": 255,
            "verbose":0
            }
    
    LGB_PARAMS = [LGB_PARAMS1, LGB_PARAMS2, LGB_PARAMS3, LGB_PARAMS4]
    
    BASE_COLS_LABEL = ['region', 
                'city', 
                'parent_category_name',
                'category_name', 
                'param_1', 
                'param_2', 
                'param_3', 
                'image_top_1',
                'user_type']
    
    BASE_COLS_TMEAN = ['user_id', 'region', 
                'city', 
                'parent_category_name',
                'category_name', 
                'param_1', 
                'param_2', 
                'param_3', 
                'image_top_1',
                'user_type']
    
    ######################   Logger   #########################################
    handler = logging.FileHandler(LOGGER_FILE)
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    ###################### Read data ##########################################
    logger.info("Reading data")
    train = pd.read_csv("../input/train.csv", parse_dates=['activation_date'], nrows=10000)
    test = pd.read_csv("../input/test.csv", parse_dates=['activation_date'], nrows=10000)
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
    # Start with base catrgoricals and their mean target encoded ones
    # Greedily check each count feature
    #
    b_thresh = 1
    b_trenc_thresh = 3
    base_cols_lbenc = [col+'_lbenc_'+str(b_thresh) for col in BASE_COLS_LABEL]
    base_cols_trenc = [col+'_trenc_'+str(b_trenc_thresh) for col in BASE_COLS_TMEAN]
    base_cols = base_cols_lbenc + base_cols_trenc
    
    columns_to_try = [col+'_counts' for col in CAT_COLS] + \
                        ["_".join(list(cols)) +'_counts' for cols in COMB_COLS]
                        
        
    features = base_cols[:]
    X = np.vstack([np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r') for col in base_cols]).T
    print("Shape for base dataset is ", X.shape)
    
    best_rmse_lgb, y_preds_lgb = eval_lgb_sets(X, y, cvlist, LGB_PARAMS)
    logger.info("Best score for base cols in {}".format(best_rmse_lgb))
    
    best_rmse = best_rmse_lgb
    y_preds_best = y_preds_lgb
    for col in columns_to_try:
        logger.info("#######################################")
        logger.info("Adding column {} and checking".format(col))
        try:
            X_col = np.load("../utility/X_train_{}.npy".format(col)).reshape(-1,1) 
            #print(X_col[:5])
            X = np.hstack((X, X_col))
            print(X.shape)
            best_rmse_lgb, y_preds_lgb = eval_lgb_sets(X, y, cvlist, LGB_PARAMS)
            
            if best_rmse_lgb < best_rmse:
                best_rmse = best_rmse_lgb
                y_preds_best = y_preds_lgb
                features.append(col)
            else:
                X = X[:, :-1]
                logger.info("{} DID NOT result in improvement".format(col))
            logger.info("")
        except:
            logger.info("Skipping {}".format(col))
            continue

    logger.info("Best score is {} with final features {}".format(best_rmse, features))
    logger.info("")
    
    
    handler.close()
    logger.removeHandler(handler)
    
    
    
    
    
    
    