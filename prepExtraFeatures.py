"""
@author - Mohsin
"""
import numpy as np
np.random.seed(786)  # for reproducibility
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

tqdm.pandas(tqdm)

from utils import *
from LabelEncodeWithThreshold import LabelEncodeWithThreshold
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    LOGGER_FILE = "prepExtraFeats.log"

    DNN_PRED_FILE = "../Avito_Kaggle/all_images_inc_xcp_res_confs.csv"
    AGG_FEAT_FILE = "../Avito_Kaggle/aggregated_features.csv"

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

    ################### Greedy forward feature selection ######################
    #
    cont_cols = ['inception_v3_prob_0', 'xception_prob_0', 'resnet50_prob_0',
                 'avg_days_up_user', 'std_days_up_user', 'avg_times_up_user', 'std_times_up_user',
                 'n_user_items', 'agg_isnull']
    cat_cols = ['inception_v3_label_0', 'xception_label_0', 'resnet50_label_0']

    EXTRA_FEATS = cont_cols + cat_cols

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


    logger.info("Saving image features 1")
    for col in EXTRA_FEATS:
        np.save("../utility/X_train_{}.npy".format(col), train_dnn_agg[col].values)
        np.save("../utility/X_test_{}.npy".format(col), test_dnn_agg[col].values)
    handler.close()
    logger.removeHandler(handler)
