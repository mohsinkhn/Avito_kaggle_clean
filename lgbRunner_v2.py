
# coding: utf-8

# In[167]:


#Author : Mohsin

import numpy as np
import pandas as pd
import os
import gc
import pickle
import string

from operator import itemgetter
from typing import List, Dict

# Models Packages
from sklearn.model_selection import cross_val_predict,  KFold
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline, make_union, Pipeline
from scipy.sparse import hstack, csr_matrix, save_npz
from nltk.corpus import stopwords 

# Gradient Boosting
import lightgbm as lgb

#Custom libraries
from LabelEncodeWithThreshold import LabelEncodeWithThreshold
from TextStatsTransformer import TextStatsTransformer
from MapFileFeatures import MapFileFeatures
from RelByCat import RelByCat
from TargetEncoder import TargetEncoder
from utils import rmse

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#%%
def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def get_col(col_name): return lambda x: x[col_name].astype(str)

def step_decay(rnd):
    if rnd < 300:
        return 0.05
    if rnd < 600:
        return 0.03
    if rnd < 1200:
        return 0.02
    else:
        return 0.01
    
def frac_decay(rnd):
    if rnd < 200:
        return 0.5
    if rnd < 400:
        return 0.4
    if rnd < 600:
        return 0.3
    else:
        return 0.25
    
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
               early_stopping_rounds=200, verbose=100, **fit_params) #Might need to change this depending on estimator
        preds[val_index] = est.predict(X_val)
        #break
        if predict_test:
            tpreds = est.predict(X_test)
            test_preds.append(tpreds)
        
    if len(test_preds) >0:
        test_preds = np.mean(test_preds, axis=0)
    return est, preds, test_preds


if __name__ == "__main__":


    #Parameters to carefully  look before running string
    MODEL_ID = "v6_iter1"
    LOGGING_FILE  = "lightgbm_{}.log".format(MODEL_ID)
    TRAIN_DATA_PATH = '../input/train.csv'
    TEST_DATA_PATH = '../input/test.csv'
    RIDGE_PRED_PATH = "./"
    ALL_IMAGE_METADATA_PATH = "./all_image_extra_feats.csv"
    PERIODS_AGG_PATH = "./aggregated_features.csv"

    LABEL_ENC_THRESH = 2
    FILL_VALUE = -0.5


    # In[2]:


    #################Set up Logging ###########################################
    # create a file handler
    handler = logging.FileHandler(LOGGING_FILE)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    ##################Read data################################################
    logger.info("\nData Load Stage")
    train = pd.read_csv(TRAIN_DATA_PATH, index_col = "item_id", parse_dates = ["activation_date"] )
    test = pd.read_csv(TEST_DATA_PATH, index_col = "item_id", parse_dates = ["activation_date"] )

    train['city'] = train['city'].astype(str) + "_" + train['region'].astype(str)
    test['city'] = test['city'].astype(str) + "_" + test['region'].astype(str)

    y = train.deal_probability.copy()
    cvlist = list(KFold(5, random_state=10, shuffle=True).split(y))
    logger.info('Train shape: {} Rows, {} Columns'.format(*train.shape))
    logger.info('Test shape: {} Rows, {} Columns'.format(*test.shape))


    # In[3]:


    ###############Punctuations and Emoji's ###################################
    logger.info("Getting punctuations and emojis")
    punct = set(string.punctuation)

    emoji = set()
    for s in train['title'].fillna('').astype(str):
        for c in s:
            if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
                continue
            emoji.add(c)

    for s in train['description'].fillna('').astype(str):
        for c in str(s):
            if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
                continue
            emoji.add(c)
    logger.info(''.join(emoji))


    ###############Load Ridge predictions #####################################
    logger.info("Loading ridge predictions")
    X_train_ridge_title = np.load(os.path.join(RIDGE_PRED_PATH, "ridge_title_oof.npy"))
    X_train_ridge_desc = np.load(os.path.join(RIDGE_PRED_PATH, "ridge_desc_oof.npy"))

    X_test_ridge_title = np.load(os.path.join(RIDGE_PRED_PATH, "ridge_title_test.npy"))
    X_test_ridge_desc = np.load(os.path.join(RIDGE_PRED_PATH, "ridge_desc_test.npy"))


    # In[213]:


    ############## Base features ##############################################
    logger.info("Processing base features")
    count_vectorizer_title = TfidfVectorizer(stop_words=stopwords.words('russian'), token_pattern="\w{1,}",
                                             lowercase=True, min_df=2, ngram_range=(1,2),
                                             )


    count_vectorizer_desc = TfidfVectorizer(stop_words=stopwords.words('russian'),
                                            lowercase=True, ngram_range=(1, 2),
                                            max_features=18000)


    categoricals =  ["user_id", "city",  "parent_category_name", "category_name",
                     "user_type","image_top_1",  "param_1", "param_2", "param_3"]

    image_feats = ['dullness', 'whiteness', 'average_pixel_width', 'blurrness',
                   'size', 'dominant_red', 'dominant_green', 'average_red',
                   'average_green', 'average_blue', 'width', 'height', 'ar']


    pipe1 = make_union(
            on_field('title', 
                     FunctionTransformer(getattr(pd.Series, 'fillna'), 
                                         validate=False, kw_args={'value':"."}), 
                     count_vectorizer_title),
            on_field('description', 
                     FunctionTransformer(getattr(pd.Series, 'fillna'), 
                                         validate=False, kw_args={'value':"."}),
                     count_vectorizer_desc),
            on_field('title',
                    TextStatsTransformer(ratios=True, puncts=punct, emojis=emoji, logger=logger)),
            on_field('description',
                    TextStatsTransformer(ratios=True, puncts=punct, emojis=emoji, logger=logger)),
            *[on_field(cat, 
                    LabelEncodeWithThreshold(thresh=LABEL_ENC_THRESH, logger=logger),
                    FunctionTransformer(getattr(pd.Series, '__array__'), validate=False),
                    FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})
                    ) 
                    for  cat in categoricals ],
            on_field('price', 
                     FunctionTransformer(getattr(pd.Series, 'fillna'), 
                                         validate=False, kw_args={'value':FILL_VALUE}),
                     FunctionTransformer(getattr(pd.Series, '__array__'), validate=False,),
                     FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)}), 
                     FunctionTransformer(np.log1p)),
            on_field('item_seq_number',
                     FunctionTransformer(getattr(pd.Series, 'fillna'), 
                                         validate=False, kw_args={'value':FILL_VALUE}),
                     FunctionTransformer(getattr(pd.Series, '__array__'), validate=False,),
                     FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)}), 
                     FunctionTransformer(np.log1p)),
            make_pipeline(MapFileFeatures(ALL_IMAGE_METADATA_PATH, map_col="image", 
                                          use_cols=image_feats, logger=logger),
                          FunctionTransformer(getattr(pd.DataFrame, 'fillna'), 
                                              validate=False, kw_args={'value':FILL_VALUE}),
                          FunctionTransformer(np.log1p)),
            make_pipeline(MapFileFeatures(PERIODS_AGG_PATH, map_col="user_id", logger=logger),
                          FunctionTransformer(getattr(pd.DataFrame, 'fillna'), 
                                              validate=False, kw_args={'value':FILL_VALUE}),
                          FunctionTransformer(np.log1p)),
        n_jobs=1,
    )


    X_train_pipe1 = pipe1.fit_transform(train)
    X_test_pipe1 = pipe1.transform(test)
    logger.info("Shape of Pipe1 feetures {} and {}".format(X_train_pipe1.shape, X_test_pipe1.shape))

    title_feats = ['title_' + f 
                   for f in pipe1.transformer_list[0][1].named_steps['tfidfvectorizer'].get_feature_names()]
    desc_feats = ["desc_" + f 
                  for f in pipe1.transformer_list[1][1].named_steps['tfidfvectorizer'].get_feature_names()]
    titles_feats = ["titles_" + f 
                    for f in pipe1.transformer_list[2][1].named_steps['textstatstransformer'].get_feature_names()]
    descs_feats = ["descs_" + f 
                   for f in pipe1.transformer_list[3][1].named_steps['textstatstransformer'].get_feature_names()]
    cnt_cat_feats = categoricals
    cont_feats = ['price', 'item_seq_number']
    imagem_feats = pipe1.transformer_list[15][1].named_steps['mapfilefeatures'].get_feature_names() 
    period_feats = pipe1.transformer_list[16][1].named_steps['mapfilefeatures'].get_feature_names() 

    pipe1_features = title_feats + desc_feats + titles_feats + descs_feats +                  cnt_cat_feats + cont_feats +  imagem_feats + period_feats


    # In[183]:


    ####################### Relative price features ###########################
    logger.info("Processing relative price  features")
    pipe2 = make_pipeline(
                make_union(
                    make_pipeline(RelByCat(cat_cols=["category_name"], rel_col="price"),
                                  FunctionTransformer(getattr(pd.Series, 'fillna'), validate=False, kw_args={'value':FILL_VALUE}),
                                  FunctionTransformer(getattr(pd.Series, '__array__'), validate=False,),
                                  FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)}),),
                    make_pipeline(RelByCat(cat_cols=["image_top_1"], rel_col="price"),
                                  FunctionTransformer(getattr(pd.Series, 'fillna'), validate=False, kw_args={'value':FILL_VALUE}),
                                  FunctionTransformer(getattr(pd.Series, '__array__'), validate=False,),
                                  FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)}),),
                    #make_pipeline(RelByCat(cat_cols=["city"], rel_col="price"),
                    #              FunctionTransformer(getattr(pd.Series, 'fillna'), validate=False, kw_args={'value':-0.5}),
                    #              FunctionTransformer(getattr(pd.Series, '__array__'), validate=False,),
                    #              FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)}),),
                    make_pipeline(RelByCat(cat_cols=["param_1"], rel_col="price"),
                                  FunctionTransformer(getattr(pd.Series, 'fillna'), validate=False, kw_args={'value':FILL_VALUE}),
                                  FunctionTransformer(getattr(pd.Series, '__array__'), validate=False,),
                                  FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)}),),
                    #make_pipeline(RelByCat(cat_cols=["category_name", "city"], rel_col="price"),
                    #              FunctionTransformer(getattr(pd.Series, 'fillna'), validate=False, kw_args={'value':FILL_VALUE}),
                    #              FunctionTransformer(getattr(pd.Series, '__array__'), validate=False,),
                    #              FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)}),),
                    #make_pipeline(RelByCat(cat_cols=["image_top_1", "city"], rel_col="price"),
                    #              FunctionTransformer(getattr(pd.Series, 'fillna'), validate=False, kw_args={'value':FILL_VALUE}),
                    #              FunctionTransformer(getattr(pd.Series, '__array__'), validate=False,),
                    #              FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)}),),
                    n_jobs=1,          
                   ), 
                  FunctionTransformer(np.log1p, validate=False,),
                  #QuantileTransformer(output_distribution='normal')
               )
    pipe2.fit(pd.concat([train, test]))
    #X_train_pipe2 = cross_val_predict(pipe2, train, y=None, cv = cvlist, method = 'transform', verbose=10, n_jobs=4)
    #X_test_pipe2 = pipe2.fit(train).transform(test)
    X_train_pipe2 = pipe2.transform(train)
    X_test_pipe2 = pipe2.transform(test)
    logger.info("Shape  of pipe 2 features {} and {}".format(X_train_pipe2.shape, X_test_pipe2.shape))

    pipe2_features = ["cat_rel_price", "image_top_rel_price",#
                      "param1_rel_price", 
                  #"cat_city_rel_price", 
                   #   "image_city_rel_price"
                     ]


    # In[233]:


    #########################Target mean features #############################
    logger.info("Processing Targt mean encoding features")
    pipe3 = make_pipeline(
                make_union(
                    make_pipeline(TargetEncoder(cols=['category_name'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['city'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['image_top_1'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['user_id'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    #make_pipeline(TargetEncoder(cols=['param_2'], targetcol='deal_probability', func='mean'),
                    #             FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['region','parent_category_name'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['city','category_name'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['city','image_top_1'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    #make_pipeline(TargetEncoder(cols=['city','user_type' ,'image_top_1'], targetcol='deal_probability', func='mean'),
                    #             FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['city','user_type' ,'category_name'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['user_type' ,'image_top_1'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['param_3' ,'image_top_1'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['param_2' ,'image_top_1'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['param_1' ,'image_top_1'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['param_1','param_2','param_3' ,'image_top_1'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['region' ,'image_top_1'], targetcol='deal_probability', func='mean'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['category_name'], targetcol='image_top_1', func='nunique'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    #make_pipeline(TargetEncoder(cols=['image_top_1'], targetcol='price', func='mean'),
                    #             FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['image_top_1'], targetcol='deal_probability', func='std'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    make_pipeline(TargetEncoder(cols=['param_1','image_top_1'], targetcol='deal_probability', func='std'),
                                 FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    #make_pipeline(TargetEncoder(cols=['param_3', 'image_top_1'], targetcol='deal_probability', func='std'),
                    #             FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
                    ), 
                 FunctionTransformer(pd.DataFrame, validate=False),           
                 FunctionTransformer(getattr(pd.DataFrame, 'fillna'), validate=False, kw_args={'value':FILL_VALUE}),
                 FunctionTransformer(np.log1p, validate=False),
                 FunctionTransformer(getattr(pd.Series, '__array__'), validate=False,)
                                )

    X_train_pipe3 = cross_val_predict(pipe3, train, y=None, cv = cvlist, method = 'transform', verbose=10, n_jobs=4)
    X_test_pipe3 = pipe3.fit(train).transform(test)
    logger.info("Pipe3 features shape {} and {}".format(X_train_pipe3.shape, X_test_pipe3.shape))

    pipe3_features = ["cat_mean", "city_mean", "image_top_mean", 
                      "uid_mean",
                  # "param_1_mean",# "param_2_mean",
                      "region_pcat_mean",
                      "city_cat_mean", 
                      "city_imaget1_mean",
                      "city_utype_cat",
                      "utype_image_top_1",
                      'p3_imaget3',
                      'p2_imaget3',
                      'p1_imaget3',
                      'p1p2p3_imaget1',
                      'region_imaget1',
                      #'imaget_price_mean',
                      'imaget_std',
                      'p1_imaget_std',
                      #'p3_imaget_std',
                     ]


    # In[184]:


    #################### Target count features ################################
    logger.info("Processing target count features")
    pipe4 = make_pipeline(make_union(
            make_pipeline(TargetEncoder(cols=['user_id'], targetcol='deal_probability', func='count'),
                         FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
            make_pipeline(TargetEncoder(cols=['city', 'image_top_1'], targetcol='deal_probability', func='count'),
                         FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
            make_pipeline(TargetEncoder(cols=['city', 'category_name'], targetcol='deal_probability', func='count'),
                         FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
            make_pipeline(TargetEncoder(cols=['image_top_1'], targetcol='deal_probability', func='count'),
                         FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
            make_pipeline(TargetEncoder(cols=['region','parent_category_name'], targetcol='deal_probability', func='count'),
                         FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
            make_pipeline(TargetEncoder(cols=['image_top_1'], targetcol='price', func='mean'),
                         FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
            #make_pipeline(TargetEncoder(cols=['param_1', 'image_top_1'], targetcol='deal_probability', func='count'),
            #             FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)}))
            make_pipeline(TargetEncoder(cols=['image_top_1'], targetcol='user_id', func='nunique'),
                         FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
            make_pipeline(TargetEncoder(cols=['image_top_1', 'city'], targetcol='user_id', func='nunique'),
                         FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
            make_pipeline(TargetEncoder(cols=['city'], targetcol='user_id', func='nunique'),
                         FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
            make_pipeline(TargetEncoder(cols=['city'], targetcol='image_top_1', func='nunique'),
                         FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)})),
            ), 
             FunctionTransformer(pd.DataFrame, validate=False),           
             FunctionTransformer(getattr(pd.DataFrame, 'fillna'), validate=False, kw_args={'value':FILL_VALUE}),
             FunctionTransformer(np.log1p, validate=False),
             FunctionTransformer(getattr(pd.Series, '__array__'), validate=False,)
                                )
    test["deal_probability"] = -1
    pipe4.fit(pd.concat([train, test])) 
    X_train_pipe4 = pipe4.transform(train)
    X_test_pipe4 = pipe4.transform(test)
    logger.info("Shape of count features {} and {}".format(X_train_pipe4.shape, X_test_pipe4.shape))

    pipe4_features = ["userid_count", 
                      "city_imaget_count", "city_cat_count", "imaget_count",
                      'region_pcat_count',#"param1_imaget_count"
                      "imaget_price_mean",
                      'cat_imaget_nunq',
                      'imaget_uid_nunq',
                      'imaget_city_uid_nunq',
                      'city_uid_nunq',
                      'city_imaget_nunq',
                     ]


    # In[155]:


    #################### Image label confidence features ######################
    logger.info("Processing image confidence feature")
    im_feats2 = ['label1_conf_mean', #'inception_v3_prob_0', 'xception_prob_0', 'resnet50_prob_0'
                ]
    pipe5 = make_pipeline(MapFileFeatures("all_images_inc_xcp_res_confs.csv", map_col="image", use_cols=im_feats2),
                          FunctionTransformer(getattr(pd.DataFrame, 'fillna'), validate=False, kw_args={'value':FILL_VALUE}),
                          #FunctionTransformer(np.reshape, validate=False, kw_args={"newshape":(-1,1)}),
                          FunctionTransformer(np.log1p))

    X_train_im2 = pipe5.fit_transform(train)
    X_test_im2 = pipe5.transform(test)


    # In[234]:


    #################### Putting it all together ##############################
    X = hstack((X_train_pipe1, X_train_pipe2, X_train_pipe3, X_train_pipe4, 
                X_train_ridge_title.reshape(-1,1), X_train_ridge_desc.reshape(-1,1),
                X_train_im2)).tocsr()

    X_test = hstack((X_test_pipe1, X_test_pipe2, X_test_pipe3, X_test_pipe4, 
                X_test_ridge_title.reshape(-1,1), X_test_ridge_desc.reshape(-1,1),
                X_test_im2)).tocsr()

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)
    logger.info("Final shape with all features {} and {}".format(X.shape, X_test.shape))

    ridge_feats = ['ridge_title', 'ridge_desc']

    feature_names = pipe1_features + pipe2_features + pipe3_features + pipe4_features + ridge_feats + im_feats2

    logger.info("Saving final features")
    save_npz("X_train_{}".format(MODEL_ID), X)
    save_npz("X_test_{}".format(MODEL_ID), X_test)
    with open("features_{}".format(MODEL_ID), "wb") as f:
        pickle.dump(feature_names, f)


    # In[236]:


    #%%
    ######### Run Model ####################################################### 
    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 127,
        'feature_fraction': 0.25,
        'bagging_fraction': .95,
        'learning_rate': 0.01,
        'max_bin':255,
        'min_data_in_leaf':200,
        'reg_alpha':1,
        'reg_lambda':1,
        'verbose': 1
    }

    logger.info("Starting model training")
    model = lgb.LGBMRegressor(n_estimators=25000)
    est, oof_preds, test_preds = cv_oof_predictions(model, X, y, cvlist, predict_test=True, 
                                               X_test=X_test, est_kwargs=lgbm_params,
                                               fit_params={"feature_name":feature_names, 
                                                'categorical_feature':categoricals,}
                                                )

    logger.info("Validation RMSE for the model is {}".format(rmse(y, oof_preds)))
    logger.info("Saving predictions")
    np.save("lgb_oof_{}.npy".format(MODEL_ID), oof_preds)
    np.save("lgb_test_{}.npy".format(MODEL_ID), test_preds)
    with open("lastfold_est_{}".format(MODEL_ID), "wb") as f:
        pickle.dump(est, f)

    sub = pd.read_csv("../input/sample_submission.csv")
    sub['deal_probability'] = np.clip(test_preds, 0, 1)
    sub['deal_probability'] = sub['deal_probability'].clip(0, 1)
    sub.to_csv("../outputs/sub_lgb_{}.csv".format(MODEL_ID), index=False)
    print(sub.head())


