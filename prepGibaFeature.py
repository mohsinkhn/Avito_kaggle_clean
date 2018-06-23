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

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    LOGGER_FILE = "prepGibaFeatures.log"

    TRAIN_GIBA_FILE = "../utility/train_proc1.csv"
    TEST_GIBA_FILE = "../utility/test_proc1.csv"

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

    ################### Read Image data #######################################
    train_giba = pd.read_csv(TRAIN_GIBA_FILE)
    test_giba = pd.read_csv(TEST_GIBA_FILE)

    del train_giba["user_id"]
    del test_giba["user_id"]
    logger.info("Features in giba file 1 are {}".format(train_giba.columns))

    #Map image data to train and test
    logger.info("Mapping Giba file 1 to train and test")
    train_giba = train.join(train_giba.set_index("item_id"), on="item_id", how="left").fillna(-1)
    test_giba = test.join(test_giba.set_index("item_id"), on="item_id", how="left").fillna(-1)

    ################### Greedy forward feature selection ######################
    #
    GIBA_FEATS = ['nchar_title', 'nchar_desc', 'titl_capE', 'titl_capR',
                   'titl_lowE', 'titl_lowR', 'titl_pun', 'desc_pun', 'titl_dig',
                   'desc_dig', 'wday', 'ce1', 'ce2', 'ce3', 'ce4', 'ce6', 'ce7',
                   'ce8', 'ce9', 'ce10', 'ce11', 'ce12', 'ce13', 'ce14', 'ce15',
                   'ce16', 'ce17', 'ce18', 'dif_time1', 'dif_isn1', 'mflag1',
                   'dif_time2', 'dif_time3', 'dif_time4', 'dif_time5', 'dif_time6',
                   'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'image1', 'image2', 'image3',
                   'image4', 'image5', 'image6', 'rev_seq']

    for col in GIBA_FEATS:
        median = train_giba[col].median()
        train_giba[col] = train_giba[col].fillna(median)
        test_giba[col] = test_giba[col].fillna(median)

    minmax = MinMaxScaler((-1, 1))
    train_giba[GIBA_FEATS] = minmax.fit_transform(train_giba[GIBA_FEATS])
    test_giba[GIBA_FEATS] = minmax.transform(test_giba[GIBA_FEATS])

    logger.info("Saving image features 1")
    for col in GIBA_FEATS:
        np.save("../utility/X_train_giba1_{}.npy".format(col), train_giba[col].values)
        np.save("../utility/X_test_giba1_{}.npy".format(col), test_giba[col].values)
    handler.close()
    logger.removeHandler(handler)
