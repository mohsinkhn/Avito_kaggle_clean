"""
@author - Mohsin
"""
import numpy as np
np.random.seed(786)  # for reproducibility
import pandas as pd
from sklearn.model_selection import KFold

from tqdm import tqdm

tqdm.pandas(tqdm)

from utils import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    LOGGER_FILE = "prepSurrogateFeatures.log"
    TRAIN_MODELS_FILE = "../utility/train_surrogate_feats.npy"
    TEST_MODELS_FILE = "../utility/test_surrogate_feats.npy"
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
        df["param_1_2_3"] = df["param_1"].astype(str) + "_" + \
                            df["param_2"].astype(str) + "_" + \
                            df["param_3"].astype(str)
    train = train.fillna(-1)
    test = test.fillna(-1)

    y = train['deal_probability'].values
    cvlist = list(KFold(5, random_state=123).split(y))

    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    #del train, test

    ################### Surrogate features #######################################
    logger.info("Map features from surrogate model outputs")
    surr_feats = ["cat_pred_labels", "utype_pred_labels", "cat_pred_probs", "param2_pred_labels"]
    train_surr = pd.DataFrame(np.load(TRAIN_MODELS_FILE), columns=surr_feats)
    test_surr = pd.DataFrame(np.load(TEST_MODELS_FILE), columns=surr_feats)

    ################### Save features                         ######################
    #
    logger.info("Saving image features 1")
    for col in surr_feats:
        np.save("../utility/X_train_{}.npy".format(col), train_surr[col].values)
        np.save("../utility/X_test_{}.npy".format(col), test_surr[col].values)
    handler.close()
    logger.removeHandler(handler)
