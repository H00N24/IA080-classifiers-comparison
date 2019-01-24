from keras.models import Sequential
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.callbacks import Callback
import functools
import time
from keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold

"""https://stackoverflow.com/a/50566908/3394494"""
def as_keras_metric(method):

    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

class KerasWrapper:

    def __init__(self, n_inputs, n_outputs, activation):
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._activation = activation
        self._create_model()

    def _create_model(self):

        self._model = Sequential()

        self._model.add(Dense(units=500, activation=self._activation,
                              input_dim=self._n_inputs))

        self._model.add(Dense(units=500, activation=self._activation))
        self._model.add(Dense(units=2000, activation=self._activation))
        self._model.add(Dense(units=self._n_outputs, activation='softmax'))

        precision = as_keras_metric(tf.metrics.precision)

        self._model.compile(loss='mean_squared_error',
                            optimizer='sgd',
                            metrics=['acc', precision])

    def _fit(self, X_train, y_train):
        return self._model.fit(X_train, y_train)

    def cross_validate(self, X, y, y_bin, splits=5):
        skf = StratifiedKFold(n_splits=splits, shuffle=True)

        fit_times = []
        train_accuracy = []
        train_precision = []

        test_accuracy = []
        test_precision = []

        for (train_index, test_index) in skf.split(X, y):

            self._model = None
            self._create_model()

            time_callback = TimeHistory()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_bin[train_index], y_bin[test_index]

            history = self._model.fit(X_train, y_train, callbacks=[time_callback])

            fit_times.append(time_callback.time)
            train_accuracy.append(history.history['acc'][-1])
            train_precision.append(history.history['precision'][-1])

            result = self._model.evaluate(X_test, y_test)

            test_accuracy.append(result[1])
            test_precision.append(result[2])

        results = {
            "fit_time": np.mean(fit_times).round(decimals=4),
            "train_accuracy": np.mean(train_accuracy).round(decimals=4),
            "train_precision": np.mean(train_precision).round(decimals=4),
            "test_accuracy": np.mean(test_accuracy).round(decimals=4),
            "test_precision": np.mean(test_precision).round(decimals=4)
        }

        return {
            self.name(): results
        }

    def name(self):
        return "NN-{}".format(self._activation)

"""https://stackoverflow.com/a/43186440/3394494"""
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.time = time.time()

    def on_train_end(self, logs={}):
        self.time = time.time() - self.time
