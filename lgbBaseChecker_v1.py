"""
@author - Mohsin
"""
import numpy as np

np.random.seed(786)  # for reproducibility
import pandas as pd
from sklearn.model_selection import KFold
from scipy.sparse import load_npz, hstack
from tqdm import tqdm
import lightgbm as lgb

tqdm.pandas(tqdm)

from utils import *
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
        if predict_test:
            tpreds = est.predict(X_test)
            test_preds.append(tpreds)
        #break
    if len(test_preds) > 0:
        test_preds = np.mean(test_preds, axis=0)
    return est, preds, test_preds #est, y_val, val_preds #


if __name__ == "__main__":
    LOGGER_FILE = "lgbBaseChecker_v1.log"
    MODEL_ID = "cleanedv1"
    NFOLDS=10,
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
    EXTRA_FEATS = ['inception_v3_prob_0', 'xception_prob_0', 'resnet50_prob_0',
                 'avg_days_up_user', 'std_days_up_user', 'avg_times_up_user', 'std_times_up_user',
                 'n_user_items', 'agg_isnull', 'inception_v3_label_0', 'xception_label_0', 'resnet50_label_0']

    EXTRA_FEATS2 = ["pcat_p123_deal_nunq", "cat_p123_deal_nunq", "city_p123_deal_nunq",
                      "region_p123_deal_nunq"]

    TEXT_PREDS = ["title_ridge_comb0", "title_lgb_comb0", "description_ridge_comb1", "description_ridge_comb0"]

    #SURR_FEATS = ["param2_pred_labels"]

    TITLE_COMB = 0
    DESC_COMB = 1

    LGB_PARAMS1 = {
            "n_estimators":15000,
            'learning_rate': 0.01,
            "num_leaves": 255,
            "colsample_bytree": 0.4,
            "subsample": 0.8,
            "reg_alpha": 0,
            "reg_lambda": 5,
            "min_data_in_leaf": 200,
            "max_bin": 512,
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
    train = pd.read_csv("../input/train.csv", parse_dates=['activation_date'])
    test = pd.read_csv("../input/test.csv", parse_dates=['activation_date'])
    test['deal_probability'] = -1

    # City correction
    for df in train, test:
        df['city'] = df['region'].astype(str) + "_" + df["city"].astype(str)
        df["param_1_2_3"] = df["param_1"].astype(str) + " " + \
                            df["param_2"].astype(str) + " " + \
                            df["param_3"].astype(str)
        df = df.fillna(-1)

    y = train['deal_probability'].values
    cvlist = list(KFold(NFOLDS, random_state=123).split(y))

    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    #del train, test

    ################### Process and load features #######################################
    logger.info("Loading text features")
    X_title = load_npz("../utility/X_train_title_{}.npz".format(TITLE_COMB))
    X_title_test = load_npz("../utility/X_test_title_{}.npz".format(TITLE_COMB))

    X_desc = load_npz("../utility/X_train_description_{}.npz".format(DESC_COMB))
    X_desc_test = load_npz("../utility/X_test_description_{}.npz".format(DESC_COMB))

    features = BASE_FEATURES + CONT_COLS + PRICE_COMB_COLS + PRICE_MEAN_COLS + COUNT_COLS + IMAGE_FEATS + GIBA_FEATS + \
               IMAGE3_FEATS + TEXT_STATS + EXTRA_FEATS + EXTRA_FEATS2 + TEXT_PREDS #+ SURR_FEATS
    X_cats = np.vstack([np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r') for col in features]).T
    X_cats_test = np.vstack([np.load("../utility/X_test_{}.npy".format(col), mmap_mode='r') for col in features]).T

    train["cat_enc"] =  np.load("../utility/X_train_category_name_lbenc_1.npy")
    test["cat_enc"] = np.load("../utility/X_test_category_name_lbenc_1.npy")
    image_top_1_cat = pd.concat([train, test]).groupby("image_top_1")["cat_enc"].apply(lambda x: x.mode())
    X_imaget1_cat = train["image_top_1"].map(image_top_1_cat).values.reshape(-1,1)
    X_imaget1_cat_test = test["image_top_1"].map(image_top_1_cat).values.reshape((-1,1))
    #pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,1), max_features=100000),
    #                      TruncatedSVD(20))
    #X_tsvd = pipe.fit_transform(train["description"].astype(str))
    #X_tsvd_test = pipe.transform(test["description"].astype(str))

    logger.info("Stack all features")
    X = hstack((X_cats, X_title, X_desc, X_imaget1_cat)).tocsr()
    X_test = hstack((X_cats_test, X_title_test, X_desc_test, X_imaget1_cat_test)).tocsr()
    logger.info("Shape for base dataset is {} and {}".format(X.shape, X_test.shape))
    ################### Run LGB ######################
    #
    feature_names = BASE_FEATURES + CONT_COLS + PRICE_COMB_COLS + PRICE_MEAN_COLS + COUNT_COLS + IMAGE_FEATS + GIBA_FEATS + \
               IMAGE3_FEATS + TEXT_STATS + EXTRA_FEATS + EXTRA_FEATS2 +  TEXT_PREDS + \
                    ["title_tfidf_"+str(i) for i in range(X_title.shape[1])] + \
                    ["desc_tfidf_" + str(i) for i in range(X_desc.shape[1])]

    categorical_features = ['region_lbenc_2', 'city_lbenc_2', 'parent_category_name_lbenc_2',
                     'category_name_lbenc_2', 'param_1_lbenc_2', 'param_2_lbenc_2', 'param_3_lbenc_2',
                     'image_top_1_lbenc_2', 'user_type_lbenc_2', 'inception_v3_label_0', 'xception_label_0',
                            'resnet50_label_0', "image3_dominant_color", "price_binned"]

    model = lgb.LGBMRegressor()
    est, y_preds_lgb, y_test_lgb = cv_oof_predictions(model, X, y, cvlist, LGB_PARAMS1, predict_test=True,
                                                      X_test=X_test,
                                                      #fit_params={"feature_name": feature_names,
                                                      #            "categorical_feature": categorical_features}
                                                      fit_params={})
    best_rmse_lgb_base = rmse(y, y_preds_lgb)
    logger.info("Best score for base cols in {}".format(best_rmse_lgb_base))

    logger.info("Saving predicitions")
    np.save("../utility/X_train_lgb_{}.npy".format(MODEL_ID), y_preds_lgb)
    np.save("../utility/X_test_lgb_{}.npy".format(MODEL_ID), y_test_lgb)

    logger.info("Writing out submissions")
    sub = pd.read_csv("../input/sample_submission.csv")
    sub['deal_probability'] = np.clip(y_test_lgb, 0, 1)
    sub['deal_probability'] = sub['deal_probability'].clip(0, 1)
    sub.to_csv("../outputs/sub_lgb_{}.csv".format(MODEL_ID), index=False)


    handler.close()
    logger.removeHandler(handler)
