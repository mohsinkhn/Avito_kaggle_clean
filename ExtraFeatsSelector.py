"""
@author - Mohsin
"""
from random import shuffle

import numpy as np

np.random.seed(786)  # for reproducibility
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import lightgbm as lgb

tqdm.pandas(tqdm)

from utils import *
from LabelEncodeWithThreshold import LabelEncodeWithThreshold
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
    return est, preds, test_preds #est, y_val, val_preds #


if __name__ == "__main__":
    LOGGER_FILE = "ExtraFeatureSelector.log"

    DNN_PRED_FILE = "../Avito_Kaggle/all_images_inc_xcp_res_confs.csv"
    AGG_FEAT_FILE = "../Avito_Kaggle/aggregated_features.csv"

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

    IMAGE_FEATS = ['image1_image_isna', 'image1_ar', 'image1_height',
                   'image1_width', 'image1_average_pixel_width',
                   'image1_average_red', 'image1_dominant_red',
                   'image1_whiteness', 'image1_dominant_green',
                   'image1_average_green', 'image1_blurrness',
                   'image1_size', 'image1_dullness', 'image1_average_blue']

    GIBA_FEATS = ['giba1_nchar_title', 'giba1_nchar_desc', 'giba1_titl_capE', 'giba1_titl_capR',
                   'giba1_titl_lowE', 'giba1_titl_lowR', 'giba1_titl_pun', 'giba1_desc_pun', 'giba1_titl_dig',
                   'giba1_desc_dig', 'giba1_wday', 'giba1_ce1', 'giba1_ce2', 'giba1_ce3', 'giba1_ce4', 'giba1_ce6', 'giba1_ce7',
                   'giba1_ce8', 'giba1_ce9', 'giba1_ce10', 'giba1_ce11', 'giba1_ce12', 'giba1_ce13', 'giba1_ce14', 'giba1_ce15',
                   'giba1_ce16', 'giba1_ce17', 'giba1_ce18', 'giba1_dif_time1', 'giba1_dif_isn1', 'giba1_mflag1',
                   'giba1_dif_time2', 'giba1_dif_time3', 'giba1_dif_time4', 'giba1_dif_time5', 'giba1_dif_time6',
                   'giba1_N1', 'giba1_N2', 'giba1_N3', 'giba1_N4', 'giba1_N5', 'giba1_N6', 'giba1_image1', 'giba1_image2', 'giba1_image3',
                   'giba1_image4', 'giba1_image5', 'giba1_image6', 'giba1_rev_seq']

    IMAGE3_FEATS = ['image3_br_std', 'image3_br_min', 'image3_sat_avg', 'image3_lum_mean',
                    'image3_lum_std', 'image3_lum_min', 'image3_contrast',
                    'image3_CF', 'image3_kp', 'image3_dominant_color',
                    'image3_dominant_color_ratio', 'image3_simplicity', 'image3_object_ratio']

    TEXT_STATS = ['title_num_emojis',
                      'title_word_len_ratio', 'title_digits_len_ratio',
                      'title_caps_len_ratio', 'title_punct_len_ratio', 'desc_num_words',
                      'desc_unq_words', 'desc_num_digits', 'desc_num_emojis',
                      'desc_word_len_ratio',
                      'desc_digits_len_ratio', 'desc_caps_len_ratio',
                      'title_num_words', 'title_unq_words', 'title_desc_word_ratio']

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

    ################### Process and load features #######################################
    logger.info("Loading deep net predictions")
    dnn_data = pd.read_csv(DNN_PRED_FILE)
    train_dnn = train.join(dnn_data.set_index("image"), on="image", how="left")
    test_dnn = test.join(dnn_data.set_index("image"), on="image", how="left")

    agg_data = pd.read_csv(AGG_FEAT_FILE)
    train_dnn_agg = train_dnn.join(agg_data.set_index("user_id"), on="user_id", how="left")
    test_dnn_agg = test_dnn.join(agg_data.set_index("user_id"), on="user_id", how="left")

    train_dnn_agg["agg_isnull"] = train_dnn_agg["n_user_items"].isnull().astype(int)
    test_dnn_agg["agg_isnull"] = test_dnn_agg["n_user_items"].isnull().astype(int)
    ################### Recurisive feature elimination ######################
    #

    features = BASE_FEATURES + CONT_COLS + PRICE_COMB_COLS + PRICE_MEAN_COLS + COUNT_COLS + IMAGE_FEATS + GIBA_FEATS + \
               IMAGE3_FEATS + TEXT_STATS
    X = np.vstack([np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r') for col in features]).T[:100000, :]
    print("Shape for base dataset is ", X.shape)
    cont_cols = ['label1_conf_mean', 'inception_v3_prob_0', 'xception_prob_0', 'resnet50_prob_0',
                 'avg_days_up_user', 'std_days_up_user', 'avg_times_up_user', 'std_times_up_user',
                 'n_user_items', 'agg_isnull']
    cat_cols = ['inception_v3_label_0', 'xception_label_0', 'resnet50_label_0']
    columns_to_try = cont_cols + cat_cols


    for col in cont_cols:
        median = train_dnn_agg[col].median()
        train_dnn_agg[col] = train_dnn_agg[col].fillna(median)
        test_dnn_agg[col] = test_dnn_agg[col].fillna(median)

        minmax = MinMaxScaler((-1, 1))
        train_dnn_agg[col] = minmax.fit_transform(train_dnn_agg[col].values.reshape(-1,1))
        test_dnn_agg[col] = minmax.transform(test_dnn_agg[col].values.reshape(-1,1))

    for col in cat_cols:
        lbenc = LabelEncodeWithThreshold(thresh=2)
        train_dnn_agg[col] = lbenc.fit_transform(train_dnn_agg[col])
        test_dnn_agg[col] = lbenc.transform(test_dnn_agg[col])

    #Run 5 times by randomly shuffling columns
    for i in range(5):
        features = BASE_FEATURES + CONT_COLS + PRICE_COMB_COLS + PRICE_MEAN_COLS + COUNT_COLS + IMAGE_FEATS + \
                   GIBA_FEATS + IMAGE3_FEATS + TEXT_STATS
        model = lgb.LGBMRegressor()
        est, y_preds_lgb, _ = cv_oof_predictions(model, X, y, cvlist, LGB_PARAMS1, predict_test=False, X_test=None,
                                                     fit_params={})
        best_rmse_lgb_base = rmse(y, y_preds_lgb)
        logger.info("Best score for base cols in {}".format(best_rmse_lgb_base))

        if i > 0:
            shuffle(columns_to_try)

        #Add all features and get base
        features_current = columns_to_try[:]
        features = BASE_FEATURES + CONT_COLS + PRICE_COMB_COLS + PRICE_MEAN_COLS + COUNT_COLS + \
                   IMAGE_FEATS + GIBA_FEATS + IMAGE3_FEATS + TEXT_STATS + features_current
        X_all = np.hstack((X, train_dnn_agg[features_current].values))
        model = lgb.LGBMRegressor()

        est, y_preds_lgb, _ = cv_oof_predictions(model, X_all, y, cvlist, LGB_PARAMS1, predict_test=False, X_test=None,
                                                 fit_params={})
        best_rmse_lgb_base = rmse(y, y_preds_lgb)
        logger.info("Best score for base + all cols in {}".format(best_rmse_lgb_base))
        best_rmse = best_rmse_lgb_base
        y_preds_best = y_preds_lgb


        for col in columns_to_try:
            #break
            #logger.info("#######################################")
            logger.info("Removing column {} and checking".format(col))
            try:
                features_current.remove(col)
                cols = [f for f in columns_to_try if f != col ]
                X_all = np.hstack((X, train_dnn_agg[features_current].values))
                print(X.shape)
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
