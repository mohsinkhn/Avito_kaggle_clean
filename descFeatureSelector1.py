"""
@author: Mohsin
"""

import gc
import itertools
from collections import Counter
import logging

import numpy as np

np.random.seed(786)  # for reproducibility
import pandas as pd

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize

from dask.multiprocessing import get

from tqdm import tqdm
import lightgbm as lgb

tqdm.pandas(tqdm)

from utils import *
from LabelEncodeWithThreshold import LabelEncodeWithThreshold
from TextCleaner import TextCleaner
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cv_oof_predictions(estimator, X, y, cvlist, est_kwargs, fit_params, predict_test=True, X_test=None, ):
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
        break
        #if predict_test:
        #    tpreds = est.predict(X_test)
        #    test_preds.append(tpreds)

    if len(test_preds) > 0:
        test_preds = np.mean(test_preds, axis=0)
    return est, y_val, val_preds #est, preds, test_preds


if __name__ == "__main__":
    LOGGER_FILE = "descFeatureSelector1.log"
    DROP_NUMBER = [True, False]
    DROP_STOPS = [True, False]
    DROP_VOWELS = [True, False]
    DO_POLYMORPH = [True, False]

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
    test = pd.read_csv("../input/test.csv", parse_dates=['activation_date'])
    test['deal_probability'] = -1

    # City correction
    for df in train, test:
        df['city'] = df['region'].astype(str) + "_" + df["city"].astype(str)
        df = df.fillna(-1)

    y = train['deal_probability'].values
    cvlist = list(KFold(5, random_state=123).split(y))

    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    #del train, test

    ################### Greedy forward feature selection ######################
    #
    model = lgb.LGBMRegressor(**LGB_PARAMS1)
    for comb in list(itertools.product(DROP_NUMBER, DROP_STOPS, DROP_VOWELS, DO_POLYMORPH)):
        drop_number, drop_stops, drop_vowels, do_polymorph = comb
        cleaner1 = TextCleaner(drop_multispaces=True,
                               drop_newline=True,
                               drop_number=drop_number,
                               drop_stopwords=drop_stops,
                               drop_vowels=drop_vowels,
                               do_polymorph=do_polymorph
                               )
        tfidf_vec = TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict',
                                    strip_accents=None, lowercase=False, preprocessor=None,
                                    tokenizer=word_tokenize, analyzer='word', stop_words=None,
                                    token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 2),
                                    max_df=1.0, min_df=30, max_features=None, vocabulary=None, binary=False,
                                    norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

        pipe = make_pipeline(cleaner1, tfidf_vec)
        X = pipe.fit_transform(train["description"].astype(str))
        logger.info("Shape of data {}".format(X.shape))
        est, y_val, y_preds = cv_oof_predictions(model, X, y, cvlist, predict_test=False,
                                                  X_test=None, est_kwargs={}, fit_params={})
        score = rmse(y_val, y_preds)
        logger.info("Score for settings {} is {}".format(comb, score))


    handler.close()
    logger.removeHandler(handler)


