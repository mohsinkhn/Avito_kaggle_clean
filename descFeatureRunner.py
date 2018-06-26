"""
@author - Mohsin
"""
import gc
import numpy as np
np.random.seed(786)  # for reproducibility
import pandas as pd

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import Ridge, SGDRegressor

import lightgbm as lgb

from scipy.sparse import load_npz
from utils import rmse, root_mean_squared_error, outoffold_crossvalidator_sparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from keras.layers import BatchNormalization, Input, GaussianDropout, Dense, Flatten, PReLU
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.base import BaseEstimator, RegressorMixin

class NNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim= None, layer_dims=[512, 256, 128], dropouts=[0.2, 0.2, 0.2],
                 batch_norm=True, batch_size = 512, epochs = 3, verbose = 1, callbacks= None, model_id="v1"):
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.dropouts = dropouts
        self.batch_norm = batch_norm
        self.callbacks = callbacks
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model_id = model_id

    def _build_model(self):
        inp = Input(shape= [self.input_dim], dtype = np.float32, sparse=True)
        x = Dense(1024)(inp)
        for dim , drop in zip(self.layer_dims, self.dropouts):
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = GaussianDropout(drop)(x)
            x = Dense(int(dim))(x)
            x = PReLU()(x)
        out = Dense(1, activation='sigmoid')(x)

        opt = Adam(lr=0.001)
        model = Model(inputs=inp, outputs=out)
        model.compile(loss=root_mean_squared_error, optimizer=opt)
        return model

    def fit(self, X, y):
        self.input_dim = int(X.shape[1])
        self.model = self._build_model()
        print(self.model.summary()  )
        X = X.astype(np.float32)
        if self.callbacks:
            self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs,
                           verbose=self.verbose,
                           callbacks=self.callbacks,
                           shuffle=True)
        else:
            self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs,
                           verbose=self.verbose,
                           shuffle=True)
        return self

    def predict(self, X, y=None):
        X = X.astype(np.float32)
        if self.model:
            if np.any(isinstance(c, ModelCheckpoint) for c in self.callbacks):
                self.model.load_weights("Model_" + str(self.model_id) + ".check")
            y_hat = self.model.predict(X, batch_size=10024)
        else:
            raise ValueError("Model not fit yet")
        return y_hat


def cv_oof_predictions(estimator, X, y, cvlist, est_kwargs, fit_params, predict_test=True, X_test=None, ):
    preds = np.zeros(len(y))  # Initialize empty array to hold prediction
    test_preds = []
    for tr_index, val_index in cvlist:
        gc.collect()
        X_tr, X_val = X[tr_index], X[val_index]
        y_tr, y_val = y[tr_index], y[val_index]
        est = estimator.set_params(**est_kwargs)
        # print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)
        est.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='rmse',
                early_stopping_rounds=50, verbose=200, **fit_params)  # Might need to change this depending on estimator
        val_preds = est.predict(X_val)

        preds[val_index] = val_preds
        print("Score for thos fold is ", score(y_val, val_preds))
        break
        if predict_test:
            tpreds = est.predict(X_test)
            test_preds.append(tpreds)

    if len(test_preds) > 0:
        test_preds = np.mean(test_preds, axis=0)
    return est, preds, test_preds


if __name__ == "__main__":
    LOGGER_FILE = "descFeatureRunner.log"

    LGB_PARAMS1 = {
            "n_estimators":10000,
            'learning_rate': 0.01,
            "num_leaves":127,
            "colsample_bytree": 0.65,
            "subsample": 0.8,
            "reg_alpha": 1,
            "reg_lambda": 1,
            "min_data_in_leaf": 10,
            "min_child_weight": 1,
            "max_bin": 255,
            "verbose":0
            }

    RIDGE_PARAMS = {
        "alpha" : 1
    }

    NN_PARAMS = {
        "layer_dims": [512, 128],
        "dropouts": [0.0, 0.0],
        "batch_size": 32,
        "epochs": 5,
        "batch_norm":True,
        "model_id": "v1"
    }
    ######################   Logger   #########################################
    handler = logging.FileHandler(LOGGER_FILE)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    ###################### Read data ##########################################
    logger.info("Reading data")
    train = pd.read_csv("../input/train.csv", usecols=["item_id", "deal_probability"])
    #test = pd.read_csv("../input/test.csv", usecols=["itemid", "deal_probability"])
            #test['deal_probability'] = -1

    y = train['deal_probability'].values
    cvlist = list(KFold(5, random_state=123).split(y))

    logger.info("Done. Read data with shape {}".format(train.shape))
    del train

    ######################Run data ################################################
    logger.info("Running all combinations")
    for i in range(2):
        logger.info("Loading features")
        X = load_npz("../utility/X_train_description_{}.npz".format(i))
        X_test = load_npz("../utility/X_test_description_{}.npz".format(i))
        logger.info("Dim of trainand test data {} and {}".format(X.shape, X_test.shape))
        #print(type(X))
        model = lgb.LGBMRegressor(**LGB_PARAMS1)
        est_lgb, y_preds_lgb, y_test_lgb = cv_oof_predictions(model, X, y, cvlist, predict_test=True,
                                                 X_test=X_test, est_kwargs={}, fit_params={})
        score = rmse(y, y_preds_lgb)
        logger.info("Score by LGB for comb {} is {}".format(i, score))
        np.save("../utility/X_train_description_lgb_comb{}".format(i), y_preds_lgb)
        np.save("../utility/X_test_description_lgb_comb{}".format(i), y_test_lgb)
        #break
        # rd = Ridge(**RIDGE_PARAMS)
        # y_preds_ridge = cross_val_predict(rd, X, y, cv=cvlist, verbose=1)
        # score = rmse(y, y_preds_ridge)
        # logger.info("Score by Ridge for comb {} is {}".format(i, score))
        # np.save("../utility/X_train_description_ridge_comb{}".format(i), y_preds_ridge)
        # y_test_ridge = rd.fit(X, y).predict(X_test)
        # np.save("../utility/X_test_description_ridge_comb{}".format(i), y_test_ridge)
        #if i == 1:
        #    nn = NNRegressor(**NN_PARAMS)
        #    y_preds_ridge = outoffold_crossvalidator_sparse(nn, X[:100000].tocsr(), y.reshape(-1,1), cvlist, check_filename="Model_"+NN_PARAMS["model_id"]+".check", multiinput=False)
        #    score = rmse(y, y_preds_ridge)
        #    logger.info("Score by Ridge for comb {} is {}".format(i, score))
            # np.save("../utility/X_train_title_lgb_comb{}".format(i), y_preds_lgb)
            # np.save("../utility/X_test_title_lgb_comb{}".format(i), y_test_lgb)
    handler.close()
    logger.removeHandler(handler)

