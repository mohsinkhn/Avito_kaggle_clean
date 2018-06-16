import numpy as np
import keras.backend as K
import gc
from sklearn import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import copy


def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

rmse_sklearn = metrics.make_scorer(rmse, greater_is_better=False)


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
    
class RMSEEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=5000)
            score = -1*rmse(self.y_val, y_pred)
            logs["val_score"] = score
            print("\n RMSE - epoch: %d - score: %.6f \n" % (epoch+1, score))


def shuffle_crossvalidator(model, X, y, cvlist, callbacks=[], X_test=None, predict_test=False, 
                           scorer = rmse, check_filename='tmp', multiinput=True, multioutput=False):
    y_trues = []
    y_preds = []
    scores = []
    y_test_preds = []
    for tr_index, val_index in cvlist:
        calls = copy.copy(callbacks)
        if multiinput:
            X_tr, X_val = [x[tr_index, :] for x in X], [x[val_index, :] for x in X]
        else:
            X_tr, X_val = X[tr_index, :], X[val_index, :]   
            
        if multioutput:
            y_tr, y_val = [yy[tr_index, :] for yy in y], [yy[val_index, :] for yy in y]
        else:
            y_tr, y_val = y[tr_index, :], y[val_index, :]  
        
        RMSE = RMSEEvaluation(validation_data=(X_val, y_val), interval=1)
            
        checkPoint = ModelCheckpoint(check_filename, monitor='val_score', save_best_only=True,
                                     save_weights_only=True, verbose=1, mode='max')
        earlystop = EarlyStopping(monitor="val_score", patience=3, mode="max")
        
        calls.append(RMSE) 
        calls.append(checkPoint)
        calls.append(earlystop)
        
        model.set_params(**{'callbacks':calls})
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        
        if predict_test:
            y_test_preds.append(model.predict(X_test))
        score = scorer(y_val, y_pred)
        scores.append(score)
        print("Score for this fold is ", score)
        y_trues.append(y_val)
        y_preds.append(y_pred)
        K.clear_session()
        gc.collect()
        #break
    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    if predict_test:
        y_test_preds = np.mean(y_test_preds, axis=0)
    score = scorer(y_trues, y_preds)
    print("Overall score on n fold CV is {}".format(score))
    
    return y_preds, y_trues, y_test_preds


def outoffold_crossvalidator(model, X, y, cvlist, callbacks=[], X_test=None, predict_test=False, 
                           scorer = rmse, check_filename='tmp', multiinput=True, multioutput=False):
    if multioutput:
        y_preds = np.zeros(y[0].shape)
    else:
        y_preds = np.zeros(y.shape)
        
    y_test_preds = []
    for tr_index, val_index in cvlist:
        calls = copy.copy(callbacks)
        if multiinput:
            X_tr, X_val = [x[tr_index, :] for x in X], [x[val_index, :] for x in X]
        else:
            X_tr, X_val = X[tr_index, :], X[val_index, :]   
            
        if multioutput:
            y_tr, y_val = [yy[tr_index, :] for yy in y], [yy[val_index, :] for yy in y]
        else:
            y_tr, y_val = y[tr_index, :], y[val_index, :]  
        
        RMSE = RMSEEvaluation(validation_data=(X_val, y_val), interval=1)
        checkPoint = ModelCheckpoint(check_filename, monitor='val_score', save_best_only=True,
                                     save_weights_only=True, verbose=1, mode='max')
        earlystop = EarlyStopping(monitor="val_score", patience=3, mode="max")
        
        calls.append(RMSE) 
        calls.append(checkPoint)
        calls.append(earlystop)
        
        model.set_params(**{'callbacks':calls})
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        if predict_test:
            y_test_preds.append(model.predict(X_test))
        print("Score for this fold is ", scorer(y_val, y_pred))
        y_preds[val_index] = y_pred
        K.clear_session()
        gc.collect()
        
    if predict_test:
        y_test_preds = np.mean(y_test_preds, axis=0)
    score = scorer(y, y_preds)
    print("Overall score on n fold CV is {}".format(score))
    
    return y_preds, y, y_test_preds


from sklearn.base import BaseEstimator, TransformerMixin
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def transform(self, X):
        return X[self.key]
    
    def fit(self, X, y = None):
        return self