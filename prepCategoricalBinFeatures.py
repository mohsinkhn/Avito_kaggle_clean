#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:38:12 2018

@author: Ajay
"""

from __future__ import print_function

import os
import re
import gc
import itertools
from collections import Counter
import logging

import pandas as pd
import numpy as np
import random
random.seed(786)
np.random.seed(786)

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, QuantileTransformer, MinMaxScaler, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn import metrics

from nltk.tokenize import word_tokenize

import dask.dataframe as dd
from dask.multiprocessing import get

from tqdm import tqdm
tqdm.pandas(tqdm)

from TargetEncoder import TargetEncoder
from ContinousBinning import ContinousBinning
from utils import *

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == '__main__':

    ############################################################
    ##   Set parameters for generating categorical features   ##
    ############################################################

    LOGGER_FILE = 'prepCategoricalBinFeatures.log'

    CAT_COLS = ['user_id',
                'region',
                'city',
                'parent_category_name',
                'category_name',
                'image_top_1',
                'param_1',
                'param_2',
                'param_3',
                'user_type']
    
    COMB_COLS = [('region', 'parent_category_name'),
                ('region', 'param_1'),
                ('region', 'category_name'),
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
                ('category_name', 'user_type'),
                ('category_name', 'image_top_1'),
                ('param_1', 'param_2'),
                ('param_1', 'param_3'),
                ('param_1', 'user_type'),
                ('param_1', 'image_top_1'),
                ('param_2', 'param_3'),
                ('param_2', 'image_top_1'),
                ('param_3', 'image_top_1'),
                ('user_type', 'image_top_1'),
                ('region', 'category_name', 'param_1'),
                ('user_type', 'region', 'parent_category_name'),
                ('user_type', 'region', 'category_name', 'param_1'),
                ('user_type', 'region', 'category_name', 'param_1', 'param_2'),
                ('user_type', 'city', 'category_name', 'param_1'),
                ('user_type', 'city', 'category_name', 'param_1', 'param_2')]
    NFOLDS = 5

    handler = logging.FileHandler(LOGGER_FILE)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    #####################################################
    ##                   Read Data                     ##
    #####################################################

    logger.info('Reading data')
    train = pd.read_csv('../input/train.csv', parse_dates = ['activation_date'])
    test = pd.read_csv('../input/test.csv', parse_dates = ['activation_date'])
    test['deal_probability'] = -1

    train['image_top_1'] = train['image_top_1'].astype(str)
    test['image_top_1'] = test['image_top_1'].astype(str)

    # city correction
    for df in train, test:
        df['city'] = df['region'].astype(str)+'_'+df['city'].astype(str)
        df = df.fillna(-1)

    y = train['deal_probability'].values
    cvlist = list(KFold(n_splits = NFOLDS, random_state = 123).split(y))

    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))

    #########################################################
    ##      Creating custom bins on deal_probability       ##
    #########################################################
    logger.info('Creating custom bins for deal probability')

    train['deal_label'] = ContinousBinning(bin_array= [-1, 0.05, 0.4, 0.7, 1.5]).fit_transform(train['deal_probability'])
    lb = LabelEncoder()
    train['deal_label'] = lb.fit_transform(train['deal_label'])

    #########################################################
    ##  Target Bin count encoding for Categorical Features ##
    #########################################################
    logger.info('Generating bin count for Categorical Features')

    for cols in CAT_COLS:
        cat_deal_count = TargetEncoder(cols = [cols], targetcol='deal_label', func = np.bincount, func_kwargs={'minlength': 4})
        try:
            X_cat_bins = cross_val_predict(cat_deal_count, train, y = train['deal_label'], cv = cvlist, method = 'transform')/int((1-1/NFOLDS)*len(train))
            X_cat_bins_test = cat_deal_count.fit(train).transform(test)/len(train)

            X_cat_bins_final = np.zeros((len(X_cat_bins), 4))
            for i in range(len(X_cat_bins)):
                if np.isnan(X_cat_bins[i]).any():
                    continue
                else:
                    X_cat_bins_final[i, 0] = X_cat_bins[i][0]
                    X_cat_bins_final[i, 1] = X_cat_bins[i][1]
                    X_cat_bins_final[i, 2] = X_cat_bins[i][2]
                    X_cat_bins_final[i, 3] = X_cat_bins[i][3]
            
            X_cat_bins_final_test = np.zeros((len(X_cat_bins_test), 4))
            for i in range(len(X_cat_bins_test)):
                if np.isnan(X_cat_bins_test[i]).any():
                    continue
                else:
                    X_cat_bins_final_test[i, 0] = X_cat_bins_test[i][0]
                    X_cat_bins_final_test[i, 1] = X_cat_bins_test[i][1]
                    X_cat_bins_final_test[i, 2] = X_cat_bins_test[i][2]
                    X_cat_bins_final_test[i, 3] = X_cat_bins_test[i][3]
            
            logger.info(f'Saving Target bin features for {cols}')

            np.save(f'../utility/X_train_{cols}_bin_count.npy', X_cat_bins_final)
            np.save(f'../utility/X_test_{cols}_bin_count.npy', X_cat_bins_final_test)
        except:
            logger.info(f'Not a valid Transformer for {cols}')
            continue
    
    #########################################################
    ##  Target Bin count encoding for Combination Features ##
    #########################################################

    logger.info('Generating bin count for Combination Features')

    for cols in COMB_COLS:
        col = '_'.join(cols)
        cat_deal_count = TargetEncoder(cols= list(cols), targetcol='deal_label', func= np.bincount, func_kwargs={'minlength': 4})
        try:
            X_cat_bins = cross_val_predict(cat_deal_count, train, y = train['deal_label'], cv = cvlist, method = 'transform')/int((1-1/NFOLDS)*len(train))
            X_cat_bins_test = cat_deal_count.fit(train).transform(test)/len(train)
            
            X_cat_bins_final = np.zeros((len(X_cat_bins), 4))
            for i in range(len(X_cat_bins)):
                if np.isnan(X_cat_bins[i]).any():
                    continue
                else:
                    X_cat_bins_final[i, 0] = X_cat_bins[i][0]
                    X_cat_bins_final[i, 1] = X_cat_bins[i][1]
                    X_cat_bins_final[i, 2] = X_cat_bins[i][2]
                    X_cat_bins_final[i, 3] = X_cat_bins[i][3]
            
            X_cat_bins_final_test = np.zeros((len(X_cat_bins_test), 4))
            for i in range(len(X_cat_bins_test)):
                if np.isnan(X_cat_bins_test[i]).any():
                    continue
                else:
                    X_cat_bins_final_test[i, 0] = X_cat_bins_test[i][0]
                    X_cat_bins_final_test[i, 1] = X_cat_bins_test[i][1]
                    X_cat_bins_final_test[i, 2] = X_cat_bins_test[i][2]
                    X_cat_bins_final_test[i, 3] = X_cat_bins_test[i][3]

            logger.info(f'Saving Taget bin encoding features for {col}')

            np.save(f'../utility/X_train_{col}_bin_count.npy', X_cat_bins_final)
            np.save(f'../utility/X_test_{col}_bin_count.npy', X_cat_bins_final_test)
        except:
            logger.info(f'Not a valid Transformer for {col}')
            continue
    
    handler.close()
    logger.removeHandler(handler)


