import numpy as np
np.random.seed(786)  # for reproducibility
from sklearn.base import BaseEstimator, RegressorMixin

from keras import Model
from keras.layers import Embedding, BatchNormalization, Dense, CuDNNGRU, Dot, Conv1D, Flatten, AveragePooling1D, MaxPooling1D, Input, Dropout,PReLU, concatenate, GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D, LeakyReLU
from keras.optimizers import Adam, RMSprop, Nadam, SGD
from keras.callbacks import ModelCheckpoint

from utils import root_mean_squared_error

from Attention import Attention


class nnRegressorGRU(BaseEstimator, RegressorMixin):
    def __init__(self,
                 input_names = [],
                 input_dims = [],
                 input_types=[],
                 hidden_kwargs = {},
                 input_pooling = [],
                 input_rnn_kwargs = [],
                 fc_list = [],
                 out_dim = 1,
                 out_kwargs = {},
                 optimizer = "adam",
                 opt_kwargs = {},
                 loss = root_mean_squared_error,
                 callbacks = {},
                 batch_size = 1024,
                 epochs = 1,
                 verbose = 1,
                 model_file = "model_v1.check"
                 ):

            self.input_names = input_names
            self.input_dims = input_dims
            self.input_types  = input_types
            self.hidden_kwargs = hidden_kwargs
            self.input_pooling = input_pooling
            self.input_rnn_kwargs = input_rnn_kwargs
            self.fc_list = fc_list
            self.out_dim = out_dim
            self.out_kwargs = out_kwargs
            self.optimizer = optimizer
            self.opt_kwargs = opt_kwargs
            self.loss = loss
            self.callbacks = callbacks
            self.batch_size = batch_size
            self.epochs = epochs
            self.verbose = verbose
            self.model_file = model_file

    def __fc_block(self, x, drop_layer, dense_layer, activation_layer, bnorm_layer=None):
        if bnorm_layer:
            x = bnorm_layer(x)
        x = drop_layer(x)
        x = dense_layer(x)
        x = activation_layer(x)
        return x

    def __first_block(self, inp_type, inp_layer, hidden_layer, pooling_layer=None, rnn_layer=None):
        if inp_type == "dense":
            if hidden_layer:
                bnorm_layer = BatchNormalization()(inp_layer)
                dense_layer = hidden_layer(bnorm_layer)
                emb = PReLU()(dense_layer)
            else:
                return inp_layer
        else:
            if hidden_layer:
                emb = hidden_layer(inp_layer)
            if rnn_layer:
                emb = rnn_layer(emb)
            if pooling_layer:
                emb = pooling_layer(emb)
            else:
                return inp_layer
        return emb

    def __prep_input(self, inps):
        inp_name, inp_dim, inp_type, hidden_kwargs, inp_pooling, rnn_kwargs = inps
        inp_layer = Input(shape=(inp_dim,), name =  inp_name+ "_inp")
        if inp_type == "embedding":
            hidden_layer = Embedding(**hidden_kwargs, name=inp_name + "_emb")
        elif inp_type == "dense":
            hidden_layer = Dense(**hidden_kwargs, name=inp_name + "_dense")
        else:
            hidden_layer = None

        if inp_pooling:
            if inp_pooling == "attention":
                pooling_layer = Attention(inp_dim)
            elif inp_pooling == "globalaverage":
                pooling_layer = GlobalAveragePooling1D()
            elif inp_pooling == "globalmax":
                pooling_layer = GlobalMaxPooling1D()
            elif inp_pooling == "flatten":
                pooling_layer = Flatten()
            else:
                pooling_layer = Flatten()
        else:
            pooling_layer = None
        if rnn_kwargs:
            rnn_layer = CuDNNGRU(**rnn_kwargs, return_sequences=True)
        else:
            rnn_layer = None
        emb = self.__first_block(inp_type, inp_layer, hidden_layer, pooling_layer, rnn_layer)
        return inp_layer, emb

    def __prep_fc(self, x, inps):
        drop_rate, dense_dim, act_layer_type, batch_norm_flag = inps
        if batch_norm_flag:
            bnorm_layer = BatchNormalization()
        else:
            bnorm_layer = None

        if drop_rate:
            drop_layer = Dropout(drop_rate)
        else:
            drop_layer = Dropout(0.0)

        dense_layer = Dense(dense_dim, kernel_initializer="glorot_uniform")

        if act_layer_type == "leakyrelu":
            activation_layer = LeakyReLU()
        else:
            activation_layer = PReLU()
        x = self.__fc_block(x, drop_layer, dense_layer, activation_layer, bnorm_layer)
        return x

    def __build_model(self):

        self.input_layers = []
        self.concat_layers = []
        for inps in zip(self.input_names, self.input_dims, self.input_types, self.hidden_kwargs,
                        self.input_pooling, self.input_rnn_kwargs):
            inp_layer, emb = self.__prep_input(inps)
            self.input_layers.append(inp_layer)
            self.concat_layers.append(emb)

        x = concatenate(self.concat_layers)

        for inps in self.fc_list:
            x = self.__prep_fc(x, inps)

        if self.out_kwargs:
            out = Dense(self.out_dim, **self.out_kwargs)(x)
        else:
            out = Dense(self.out_dim)(x)

        if self.optimizer == 'adam':
            opt = Adam(**self.opt_kwargs)
        if self.optimizer == 'nadam':
            opt = Nadam(**self.opt_kwargs)
        if self.optimizer == 'sgd':
            opt = SGD(**self.opt_kwargs)
        elif self.optimizer == 'rmsprop':
            opt = RMSprop(**self.opt_kwargs)

        model = Model(inputs=self.input_layers, outputs=out)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model

    def fit(self, X, y):
        self.model = self.__build_model()

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
        if self.model:
            if np.any(isinstance(c, ModelCheckpoint) for c in self.callbacks):
                self.model.load_weights(self.model_file)
            y_hat = self.model.predict(X, batch_size=10024)
        else:
            raise ValueError("Model not fit yet")
        return y_hat




