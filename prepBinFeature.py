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
    LOGGER_FILE = "prepBinFeature.log"

    BIN_COUNT_FEATS = ['user_type_region_category_name_param_1_bin_count', 'region_param_1_bin_count','user_type_region_category_name_param_1_param_2_bin_count']

    ######################   Logger   #########################################
    handler = logging.FileHandler(LOGGER_FILE)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    ###################### Read data ##########################################
    logger.info("Reading data")
    

    for feat in BIN_COUNT_FEATS:
        temp = np.load(f'../utility/X_train_{feat}.npy')
        temp1 = np.load(f'../utility/X_test_{feat}.npy')
        for i in range(temp.shape[1]):
            name = feat+str(i)
            np.save(f'../utility/X_train_{name}.npy', temp[:, 0])
            np.save(f'../utility/X_test_{name}.npy', temp1[:, 0])
    
    handler.close()
    logger.removeHandler(handler)
