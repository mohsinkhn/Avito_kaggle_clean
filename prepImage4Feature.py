import pandas as pd
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
    LOGGER_FILE = "prepImageFeature4.log"

    IMAGE_FILE_1 =  "../utility/df_imgtop1_preds.csv"

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
    
    train = train.fillna(-1)
    test = test.fillna(-1)

    y = train['deal_probability'].values
    cvlist = list(KFold(5, random_state=123).split(y))

    logger.info("Done. Read data with shape {} and {}".format(train.shape, test.shape))
    #del train, test

    ################### Read Image data #######################################
    image_df_1 = pd.read_csv(IMAGE_FILE_1)
    logger.info("Features in image file 4 are {}".format(image_df_1.columns))

    #Create difference between image_top_1 pred and actual

    image_df_1['diff_top_1_label'] = image_df_1['imt_top1_label'] - image_df_1['image_top_1']
    image_df_1['diff_top_2_label'] = image_df_1['imt_top2_label'] - image_df_1['image_top_1']
    image_df_1['diff_top_3_label'] = image_df_1['imt_top3_label'] - image_df_1['image_top_1']

    IMAGE_FEATS = ['imt_top1_prob', 'imt_top2_prob',
                    'imt_top3_prob', 'diff_top_1_label', 'diff_top_2_label', 'diff_top_3_label']

    #Map image data to train and test
    logger.info("Mapping image file 3 to train and test")
    train_img = train.join(image_df_1[IMAGE_FEATS + ['image']].set_index("image"), on="image", how="left")
    test_img = test.join(image_df_1[IMAGE_FEATS + ['image']].set_index("image"), on="image", how="left")

    #train_img["image_isna"] = train_img["size"].isnull().astype(int)
    #test_img["image_isna"] = test_img["size"].isnull().astype(int)
    ################### Greedy forward feature selection ######################
    #
    
    for col in IMAGE_FEATS:
        if '_prob' in col:
            temp = train_img[col].min() - 0.1*(train_img[col].max() - train_img[col].min())
            train_img[col] = train_img[col].fillna(temp)
            test_img[col] = test_img[col].fillna(temp)
        else:
            train_img[col] = train_img[col].fillna(-1)
            test_img[col] = test_img[col].fillna(-1)

    minmax = MinMaxScaler((-1, 1))
    #minmax = FunctionTransformer(np.log1p, validate=False)
    #minmax = QuantileTransformer(output_distribution="normal")
    diff_feats = [f for f in IMAGE_FEATS if 'diff_' in f]
    train_img[diff_feats] = minmax.fit_transform(train_img[diff_feats])
    test_img[diff_feats] = minmax.transform(test_img[diff_feats])


    logger.info("Saving image features 1")
    for col in IMAGE_FEATS:
        np.save("../utility/X_train_image4_{}.npy".format(col), train_img[col].values)
        np.save("../utility/X_test_image4_{}.npy".format(col), test_img[col].values)
    handler.close()
    logger.removeHandler(handler)
