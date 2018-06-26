from __future__ import print_function
import numpy as np
from copy import copy
np.random.seed(786)  # for reproducibility
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from keras import regularizers
from utils import root_mean_squared_error, outoffold_crossvalidator, rmse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from nnRgressorGRU import nnRegressorGRU

if __name__=="__main__":

    #########################################################
    ##  Set Parameters for NN ##
    #########################################################
    LOGGER_FILE = "logs/nnDesc_trainerv3.log"
    MODEL_ID = "desc_v3"
    MODEL_CHECK_FILENAME = "{}.check".format(MODEL_ID)
    DESC_NAMES = ["desctokens_2",
                   "desc2gramtokens_0"
                 ]

    #Settings for categoricals
    DESC_DIMS = [80, 20]
    DESC_TYPES = ["embedding"] * 2
    DESC_EMBED_KWARGS = [{"input_dim": 50000, "output_dim": 32,
                           "input_length": 80, },
                          {"input_dim": 10000, "output_dim": 12,
                           "input_length": 20, }
                          ]
    DESC_RNN_KWARGS = [None, None]
    DESC_POOLING = ["attention"] * 2

    #Settings for Continous
    #CONTS = ["price", "item_seq_number", "image_top_1_cont", "user_id_trenc_1"]
    #COUNTS = ['user_id_counts', 'param_1_counts', 'user_type_counts', 'param_1_user_type_activation_date_counts',
    #              'param_2_user_type_activation_date_counts',
    #              'region_param_1_user_type_activation_date_counts',
    #              'city_category_name_param_1_user_type_counts',
    #              'city_category_name_param_2_user_type_counts']
    #CONT_NAMES = CONTS + COUNTS
    #CONT_DIMS = [1] * (len(CONT_NAMES))
    #CONT_TYPES = ["dense"] * (len(CONT_NAMES))
    #CONT_DENSE_DIMS = [24, 16, 16, 16] + [12] * len(COUNTS)
    #CONT_DENSE_KWARGS = [{"units": units, "kernel_initializer": "glorot_normal"} for units in CONT_DENSE_DIMS]
    #CONT_RNN_KWARGS = [None] * (len(CONT_NAMES))
    #CONT_POOLING = [None] * (len(CONT_NAMES))

    FC1_PARAMS = [0.0, 44, "prelu", False]
    FC2_PARAMS = [0.0, 128, "prelu", True]
    FC3_PARAMS = [0.0, 128, "prelu", True]

    INPUT_NAMES = DESC_NAMES #CAT_NAMES + CONT_NAMES
    INPUT_DIMS =  DESC_DIMS#CAT_DIMS + CONT_DIMS
    INPUT_TYPES = DESC_TYPES#CAT_TYPES + CONT_TYPES
    HIDDEN_KWARGS = DESC_EMBED_KWARGS#CAT_EMBED_KWARGS + CONT_DENSE_KWARGS
    INPUT_RNN_KWARGS = DESC_RNN_KWARGS#CAT_RNN_KWARGS + CONT_RNN_KWARGS
    INPUT_POOLING = DESC_POOLING#CAT_POOLING + CONT_POOLING
    FC_LIST = [FC1_PARAMS]

    NNREG = nnRegressorGRU( input_names=INPUT_NAMES,
                            input_dims=INPUT_DIMS,
                            input_types=INPUT_TYPES,
                            hidden_kwargs=HIDDEN_KWARGS,
                            input_rnn_kwargs=INPUT_RNN_KWARGS,
                            input_pooling=INPUT_POOLING,
                            fc_list=FC_LIST,
                            out_dim=1,
                            out_kwargs={"activation": "sigmoid", "kernel_initializer":"glorot_uniform"},
                            loss=root_mean_squared_error,
                            batch_size=800,
                            optimizer="adam",
                            opt_kwargs={"lr": 0.001, "decay": 0.001},
                            epochs=2,
                            verbose=1,
                            model_file=MODEL_CHECK_FILENAME)

    ############ Setup logger ###############################
    handler = logging.FileHandler(LOGGER_FILE)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    #########################################################
    ##  Read data                                          ##
    #########################################################
    logger.info("Reading target data")

    y = pd.read_csv("../input/train.csv", usecols=['deal_probability']).values.reshape(-1,1)
    cvlist = list(KFold(5, random_state=123).split(y))

    X_train = []
    X_test = []

    for col in DESC_NAMES:
        arr = np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r')
        X_train.append(arr)

        arr = np.load("../utility/X_test_{}.npy".format(col), mmap_mode='r')
        X_test.append(arr)

    #for col in CAT_NAMES:
    #    arr = np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
    #    X_train.append(arr)

    #    arr = np.load("../utility/X_test_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
    #    X_test.append(arr)

    #scaler = QuantileTransformer(output_distribution="normal")
    #for col in CONTS:
    #    tmp = np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
    #    tmp = scaler.fit_transform(tmp)
    #    X_train.append(tmp)

    #    tmp = np.load("../utility/X_test_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
    #    tmp = scaler.transform(tmp)
    #    X_test.append(tmp)

    #scaler = MinMaxScaler((-1,1))
    #for col in COUNTS:
    #    tmp = np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
    #    tmp = scaler.fit_transform(tmp)
    #    X_train.append(tmp)

    #    tmp = np.load("../utility/X_test_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
    #    tmp = scaler.transform(tmp)
    #    X_test.append(tmp)

    #########################################################
    ## Run MODEL                                    ###
    #########################################################

    #Run 3 times ans average
    y_preds_mean = []
    y_test_mean = []
    for i in range(3):
        y_preds, y_trues, y_test = outoffold_crossvalidator(NNREG, X_train,
                                                        y, cvlist, check_filename=MODEL_CHECK_FILENAME, logger=LOGGER_FILE)

        logger.info("Val RMSE for {}th iteration is {}".format(i, rmse(y_trues, y_preds)))
        y_preds_mean.append(y_preds)
        y_test_mean.append(y_test)
        #break
    y_preds_mean = np.mean(y_preds_mean, axis=0)
    y_test_mean = np.mean(y_test_mean, axis=0)

    logger.info("Val RMSE after averaging 3 iterations is {}".format(rmse(y, y_preds_mean)))
    ############Save outputs #################################
    np.save("../utility/X_train_nnpreds_{}.npy".format(MODEL_ID), y_preds_mean)
    np.save("../utility/X_test_nnpreds_{}.npy".format(MODEL_ID), y_test_mean)