#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 08:36:31 2018

@author: mohsin
"""

from __future__ import print_function
import logging
import numpy as np
np.random.seed(786)  # for reproducibility
import pandas as pd
from tqdm import tqdm
tqdm.pandas(tqdm)

from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#%%
if __name__ == "__main__":
    
    #########################################################
    ##  Set Parameters ##
    #########################################################
    
    LOGGER_FILE = "prepImageTop1Cont.log"
    CONT_COLS = ["image_top_1_cont"]

    ############ Setup logger ###############################
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
    
    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    
    
    #########################################################
    ##  Generate feature                                   ##
    #########################################################
    cat_image_top_1 = pd.concat([train, test]).groupby("category_name")["image_top_1"].median()
    train["cat_image_top_1"] = train["category_name"].map(cat_image_top_1).fillna(0)
    test["cat_image_top_1"] = test["category_name"].map(cat_image_top_1).fillna(0)

    train.loc[train.image_top_1.isnull(), "image_top_1"] = train.loc[train.image_top_1.isnull(), "cat_image_top_1"]
    test.loc[test.image_top_1.isnull(), "image_top_1"] = test.loc[test.image_top_1.isnull(), "cat_image_top_1"]

    #########################################################
    ##  Save feature                                       ##
    #########################################################
    col_name = "image_top_1_cont"
    np.save("../utility/X_train_{}.npy".format(col_name), train["image_top_1"].values.reshape(-1,1))
    np.save("../utility/X_test_{}.npy".format(col_name),  test["image_top_1"].values.reshape(-1,1))
    logger.info("Done!")
