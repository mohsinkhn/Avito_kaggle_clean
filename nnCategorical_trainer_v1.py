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
    LOGGER_FILE = "logs/nnCategorical_trainerv1.log"
    MODEL_ID = "catnum_v1"
    MODEL_CHECK_FILENAME = "{}.check".format(MODEL_ID)
    CAT_NAMES = ["region_lbenc_1", "image_top_1_lbenc_1", "city_lbenc_1",
                 "parent_category_name_lbenc_1", "category_name_lbenc_1",
                 "param_1_lbenc_1", "param_2_lbenc_1", "param_3_lbenc_1",
                 "user_type_lbenc_1", "param_1_param_2_lbenc_3",
                #"user_id_lbenc_8"
                 ]

    #Settings for categoricals
    CAT_DIMS = [1] * len(CAT_NAMES)
    CAT_TYPES = ["embedding"] * len(CAT_NAMES)
    CAT_EMBED_INPS = [28, 1752, 3063, 9, 47, 372, 278, 1277, 3, 727]
    CAT_EMBED_OUTS = [24, 32,     32, 16, 32, 16, 6,   16, 8 , 8]
    CAT_EMBED_KWARGS = [{"input_dim": inp, "output_dim": out, "embeddings_initializer":"glorot_uniform"}
                          for inp, out in zip(CAT_EMBED_INPS, CAT_EMBED_OUTS)] #+\
                        #[{"input_dim": 584637, "output_dim": 16, "embeddings_initializer":"glorot_uniform",
                        #  "embeddings_regularizer": regularizers.l2(1e-6)}]

    CAT_RNN_KWARGS = [None] * len(CAT_NAMES)
    CAT_POOLING = ["flatten"] * len(CAT_NAMES)

    #Settings for Continous
    CONTS = ["price", "item_seq_number", "image_top_1_cont", "user_id_trenc_1"]
    COUNTS = ['user_id_counts', 'param_1_counts', 'user_type_counts', 'param_1_user_type_activation_date_counts',
                  'param_2_user_type_activation_date_counts',
                  'region_param_1_user_type_activation_date_counts',
                  'city_category_name_param_1_user_type_counts',
                  'city_category_name_param_2_user_type_counts']
    CONT_NAMES = CONTS + COUNTS
    CONT_DIMS = [1] * (len(CONT_NAMES))
    CONT_TYPES = ["dense"] * (len(CONT_NAMES))
    CONT_DENSE_DIMS = [24, 16, 16, 16] + [12] * len(COUNTS)
    CONT_DENSE_KWARGS = [{"units": units, "kernel_initializer": "glorot_normal"} for units in CONT_DENSE_DIMS]
    CONT_RNN_KWARGS = [None] * (len(CONT_NAMES))
    CONT_POOLING = [None] * (len(CONT_NAMES))

    FC1_PARAMS = [0.0, 300, "prelu", True]
    FC2_PARAMS = [0.0, 128, "prelu", True]
    FC3_PARAMS = [0.0, 128, "prelu", True]

    INPUT_NAMES = CAT_NAMES + CONT_NAMES
    INPUT_DIMS = CAT_DIMS + CONT_DIMS
    INPUT_TYPES = CAT_TYPES + CONT_TYPES
    HIDDEN_KWARGS = CAT_EMBED_KWARGS + CONT_DENSE_KWARGS
    INPUT_RNN_KWARGS = CAT_RNN_KWARGS + CONT_RNN_KWARGS
    INPUT_POOLING = CAT_POOLING + CONT_POOLING
    FC_LIST = [FC1_PARAMS, FC2_PARAMS, FC3_PARAMS]

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
                            opt_kwargs={"lr": 0.001},
                            epochs=10,
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
    for col in CAT_NAMES:
        arr = np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
        X_train.append(arr)

    for col in CAT_NAMES:
        arr = np.load("../utility/X_test_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
        X_test.append(arr)

    scaler = QuantileTransformer(output_distribution="normal")
    for col in CONTS:
        tmp = np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
        tmp = scaler.fit_transform(tmp)
        X_train.append(tmp)

    for col in CONTS:
        tmp = np.load("../utility/X_test_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
        tmp = scaler.transform(tmp)
        X_test.append(tmp)

    scaler = MinMaxScaler((-1,1))
    for col in COUNTS:
        tmp = np.load("../utility/X_train_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
        tmp = scaler.fit_transform(tmp)
        X_train.append(tmp)

    for col in COUNTS:
        tmp = np.load("../utility/X_test_{}.npy".format(col), mmap_mode='r').reshape(-1,1)
        tmp = scaler.transform(tmp)
        X_test.append(tmp)

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

    y_preds_mean = np.mean(y_preds_mean, axis=0)
    y_test_mean = np.mean(y_test_mean, axis=0)

    logger.info("Val RMSE after averaging 3 iterations is {}".format(rmse(y, y_preds_mean)))
    ############Save outputs #################################
    np.save("../utility/X_train_nnpreds_{}.npy".format(MODEL_ID), y_preds_mean)
    np.save("../utility/X_test_nnpreds_{}.npy".format(MODEL_ID), y_test_mean)