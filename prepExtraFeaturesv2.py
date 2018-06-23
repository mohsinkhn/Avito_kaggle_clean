"""
@author - Mohsin
"""
import numpy as np
np.random.seed(786)  # for reproducibility
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from tqdm import tqdm

tqdm.pandas(tqdm)

from utils import *
from TargetEncoder import TargetEncoder
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    LOGGER_FILE = "prepExtraFeatsv2.log"

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
        df = df.fillna(-1)

    y = train['deal_probability'].values
    cvlist = list(KFold(5, random_state=123).split(y))

    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    #del train, test

    ################### Unique features #######################################
    logger.info("Get unique counts for different combinations")
    trenc = TargetEncoder(cols=["parent_category_name", "param_1_2_3"],
                          targetcol = "deal_probability",
                          func= 'nunique')
    train["pcat_p123_deal_nunq"] = cross_val_predict(trenc, train, y, cv=cvlist, method='transform')
    test["pcat_p123_deal_nunq"] = trenc.fit(train).transform(test)

    trenc = TargetEncoder(cols=["category_name", "param_1_2_3"],
                          targetcol = "deal_probability",
                          func= 'nunique')
    train["cat_p123_deal_nunq"] = cross_val_predict(trenc, train, y, cv=cvlist, method='transform')
    test["cat_p123_deal_nunq"] = trenc.fit(train).transform(test)

    trenc = TargetEncoder(cols=["city", "param_1_2_3"],
                          targetcol = "deal_probability",
                          func= 'nunique')
    train["city_p123_deal_nunq"] = cross_val_predict(trenc, train, y, cv=cvlist, method='transform')
    test["city_p123_deal_nunq"] = trenc.fit(train).transform(test)

    trenc = TargetEncoder(cols=["region", "param_1_2_3"],
                          targetcol = "deal_probability",
                          func= 'nunique')
    train["region_p123_deal_nunq"] = cross_val_predict(trenc, train, y, cv=cvlist, method='transform')
    test["region_p123_deal_nunq"] = trenc.fit(train).transform(test)

    ################### Save features                         ######################
    #

    EXTRA_FEATS = ["pcat_p123_deal_nunq", "cat_p123_deal_nunq", "city_p123_deal_nunq",
                      "region_p123_deal_nunq"]



    logger.info("Saving image features 1")
    for col in EXTRA_FEATS:
        np.save("../utility/X_train_{}.npy".format(col), train[col].values)
        np.save("../utility/X_test_{}.npy".format(col), test[col].values)
    handler.close()
    logger.removeHandler(handler)
