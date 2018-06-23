"""
@author - Mohsin
"""
import numpy as np
np.random.seed(786)  # for reproducibility
import pandas as pd

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import FunctionTransformer,MinMaxScaler
from tqdm import tqdm


tqdm.pandas(tqdm)
from utils import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    LOGGER_FILE = "prepImageFeature1.log"

    IMAGE_FILE_1 = "../utility/all_image_extra_feats.csv"

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
    image_df_1 = pd.read_csv(IMAGE_FILE_1)
    logger.info("Feature sin image file 1 are {}".format(image_df_1.columns))

    #Map image data to train and test
    logger.info("Mapping image file 1 to train and test")
    train_img = train.join(image_df_1.set_index("image"), on="image", how="left")
    test_img = test.join(image_df_1.set_index("image"), on="image", how="left")

    train_img["image_isna"] = train_img["size"].isnull().astype(int)
    test_img["image_isna"] = test_img["size"].isnull().astype(int)
    ################### Greedy forward feature selection ######################
    #
    IMAGE_FEATS = ['image_isna', 'ar', 'height', 'width', 'average_pixel_width', 'average_red', 'dominant_red',
                   'whiteness', 'dominant_green', 'average_green', 'blurrness', 'size', 'dullness', 'average_blue']
    for col in IMAGE_FEATS:
        median = train_img[col].median()
        train_img[col] = train_img[col].fillna(median)
        test_img[col] = test_img[col].fillna(median)

    minmax = MinMaxScaler((-1, 1))
    #minmax = FunctionTransformer(np.log1p, validate=False)
    #minmax = QuantileTransformer(output_distribution="normal")
    train_img[IMAGE_FEATS] = minmax.fit_transform(train_img[IMAGE_FEATS])
    test_img[IMAGE_FEATS] = minmax.transform(test_img[IMAGE_FEATS])


    logger.info("Saving image features 1")
    for col in IMAGE_FEATS:
        np.save("../utility/X_train_image1_{}.npy".format(col), train_img[col].values)
        np.save("../utility/X_test_image1_{}.npy".format(col), test_img[col].values)
    handler.close()
    logger.removeHandler(handler)
