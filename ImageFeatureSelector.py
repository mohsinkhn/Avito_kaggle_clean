"""
@author - Mohsin
"""
import numpy as np
np.random.seed(786)  # for reproducibility
import pandas as pd
from random import shuffle
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, QuantileTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize

import dask.dataframe as dd

from tqdm import tqdm
import lightgbm as lgb

tqdm.pandas(tqdm)

from scipy.sparse import load_npz, save_npz
from TargetEncoder import TargetEncoder
from utils import *
from LabelEncodeWithThreshold import LabelEncodeWithThreshold
from TextCleaner import TextCleaner
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def cv_oof_predictions(estimator, X, y, cvlist, est_kwargs, fit_params, predict_test=False, X_test=None, ):
    preds = np.zeros(len(y))  # Initialize empty array to hold prediction
    test_preds = []
    for tr_index, val_index in cvlist:
        gc.collect()
        X_tr, X_val = X[tr_index], X[val_index]
        y_tr, y_val = y[tr_index], y[val_index]
        est = estimator.set_params(**est_kwargs)
        # print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)
        est.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='rmse',
                early_stopping_rounds=50, verbose=200, **fit_params)  # Might need to change this depending on estimator
        val_preds = est.predict(X_val)

        preds[val_index] = val_preds
        #break
        #if predict_test:
        #    tpreds = est.predict(X_test)
        #    test_preds.append(tpreds)

    if len(test_preds) > 0:
        test_preds = np.mean(test_preds, axis=0)
    return est, preds, test_preds#est, y_val, val_preds #


if __name__ == "__main__":
    LOGGER_FILE = "lgbImageFeatureSelector.log"

    IMAGE_FILE_1 = "../Avito_Kaggle/all_image_extra_feats.csv"

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
            "colsample_bytree": 0.33,
            "subsample": 0.9,
            "reg_alpha": 0,
            "reg_lambda": 0,
            "min_data_in_leaf": 200,
            "max_bin": 255,
            "verbose":0
            }

    ######################   Logger   #########################################
    handler = logging.FileHandler(LOGGER_FILE)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    ###################### Read data ##########################################
    logger.info("Reading data")
    train = pd.read_csv("../input/train.csv", parse_dates=['activation_date'], nrows=100000)
    test = pd.read_csv("../input/test.csv", parse_dates=['activation_date'], nrows=100000)
    test['deal_probability'] = -1

    # City correction
    for df in train, test:
        df['city'] = df['region'].astype(str) + "_" + df["city"].astype(str)
        df = df.fillna(-1)

    y = train['deal_probability'].values
    cvlist = list(KFold(5, random_state=123).split(y))

    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    #del train, test

    ################### Read Image data #######################################
    image_df_1 = pd.read_csv(IMAGE_FILE_1)
    logger.info("Feature sin image file 1 are {}".format(image_df_1.columns))

    #Map image data to train and test
    logger.info("Mapping image file 1 to train and test")
    train_img = train.join(image_df_1.set_index("image"), on="image", how="left")
    test_img = test.join(image_df_1.set_index("image"), on="image", how="left")

    train_img["image_isna"] = train_img["size"].isnull().astype(int)
    test_img["image_isna"] = test_img["size"].isnull().astype(int)

    ################### Recurisive feature elimination ######################
    #

    features = BASE_FEATURES + CONT_COLS + PRICE_COMB_COLS + PRICE_MEAN_COLS + COUNT_COLS
    X = np.vstack([np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r') for col in features]).T[:100000, :]
    print("Shape for base dataset is ", X.shape)
    columns_to_try = ["image_isna" , 'ar', 'height', 'width', 'average_pixel_width',
                      'average_red', 'dominant_red',  'whiteness', 'dominant_green',
                      'average_green',  'blurrness', 'size', 'dullness',
                      'average_blue']

    for col in columns_to_try:
        median = train_img[col].median()
        train_img[col] = train_img[col].fillna(median)
        test_img[col] = test_img[col].fillna(median)

    minmax = MinMaxScaler((-1, 1))
    #minmax = FunctionTransformer(np.log1p, validate=False)
    #minmax = QuantileTransformer(output_distribution="normal")
    train_img[columns_to_try] = minmax.fit_transform(train_img[columns_to_try])
    test_img[columns_to_try] = minmax.transform(test_img[columns_to_try])

    #Run 5 times by randomly shuffling columns
    for i in range(5):
        features = BASE_FEATURES + CONT_COLS + PRICE_COMB_COLS + PRICE_MEAN_COLS + COUNT_COLS
        model = lgb.LGBMRegressor()
        #est, y_val, y_preds_lgb = cv_oof_predictions(model, X, y, cvlist, LGB_PARAMS1, predict_test=False, X_test=None,
        #                                             fit_params={})
        #best_rmse_lgb_base = rmse(y_val, y_preds_lgb)
        #est, y_preds_lgb, _ = cv_oof_predictions(model, X, y, cvlist, LGB_PARAMS1, predict_test=False, X_test=None,
        #                                             fit_params={})
        #best_rmse_lgb_base = rmse(y, y_preds_lgb)
        #logger.info("Best score for base cols in {}".format(best_rmse_lgb_base))
        #best_rmse = best_rmse_lgb_base
        #y_preds_best = y_preds_lgb

        #if i > 0:
        #    shuffle(columns_to_try)

        #Add all features and get base
        features_current = columns_to_try[:]
        features = BASE_FEATURES + CONT_COLS + PRICE_COMB_COLS + PRICE_MEAN_COLS + COUNT_COLS + features_current
        X_all = np.hstack((X, train_img[features_current].values))
        model = lgb.LGBMRegressor()
        # est, y_val, y_preds_lgb = cv_oof_predictions(model, X, y, cvlist, LGB_PARAMS1, predict_test=False, X_test=None,
        #                                             fit_params={})
        # best_rmse_lgb_base = rmse(y_val, y_preds_lgb)
        est, y_preds_lgb, _ = cv_oof_predictions(model, X_all, y, cvlist, LGB_PARAMS1, predict_test=False, X_test=None,
                                                 fit_params={})
        best_rmse_lgb_base = rmse(y, y_preds_lgb)
        logger.info("Best score for base + all cols in {}".format(best_rmse_lgb_base))
        best_rmse = best_rmse_lgb_base
        y_preds_best = y_preds_lgb


        for col in columns_to_try:
            break
            logger.info("#######################################")
            logger.info("Removing column {} and checking".format(col))
            try:
                features_current.remove(col)
                cols = [f for f in columns_to_try if f != col ]
                X_all = np.hstack((X, train_img[features_current].values))
                # print(X_col[:5])
                #X = np.hstack((X, X_col))
                print(X.shape)
                #est, y_val, y_preds_lgb = cv_oof_predictions(model, X, y, cvlist, LGB_PARAMS1, predict_test=False, X_test=None,
                #                                         fit_params={})
                #best_rmse_lgb = rmse(y_val, y_preds_lgb)
                est, y_preds_lgb, _ = cv_oof_predictions(model, X_all, y, cvlist, LGB_PARAMS1, predict_test=False, X_test=None,
                                                         fit_params={})
                best_rmse_lgb = rmse(y, y_preds_lgb)
                logger.info("Score after removing {} is {}".format(col, best_rmse_lgb))
                if best_rmse_lgb > best_rmse:
                    features_current.append(col)
                else:
                    best_rmse = best_rmse_lgb
                    y_preds_best = y_preds_lgb
                    logger.info("Removing {} resulted in improvement".format(col))
                logger.info("")
            except:
                logger.info("Skipping {}".format(col))
                continue
            logger.info("Current set of features are : {}".format(features_current))

        logger.info("Score for iter {} with set of features {} is {}".format(i, features_current, best_rmse))
        break
    handler.close()
    logger.removeHandler(handler)
