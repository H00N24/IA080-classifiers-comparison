from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import functools
import time
from keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

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
    def __init__(
        self, n_inputs, n_outputs, layers, act_hidden_layers, act_output_layer_loss
    ):
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._layers = layers
        self._act_hidden_layers = act_hidden_layers
        self._act_output_layer = act_output_layer_loss[0]
        self._name_loss = act_output_layer_loss[1]
        if self._name_loss == "crossentropy":
            if self._n_outputs == 1:
                self._loss = "binary_crossentropy"
            else:
                self._loss = "categorical_crossentropy"
        else:
            self._loss = self._name_loss
        self._create_model()

    def _create_model(self):

        self._model = Sequential()

        self._model.add(Dropout(0.5))
        for units in self._layers:
            self._model.add(
                Dense(
                    units=units,
                    activation=self._act_hidden_layers,
                    kernel_initializer="glorot_normal",
                    bias_initializer="glorot_normal",
                )
            )

        self._model.add(
            Dense(
                units=self._n_outputs,
                activation=self._act_output_layer,
                kernel_initializer="glorot_normal",
                bias_initializer="glorot_normal",
            )
        )

        precision = as_keras_metric(tf.metrics.precision)

        if self._n_outputs == 1 and self._loss == "categorical_crossentropy":
            self._loss = "binary_crossentropy"

        self._model.compile(
            loss=self._loss, optimizer="adam", metrics=["acc", precision]
        )

    def _fit(self, X_train, y_train):
        return self._model.fit(X_train, y_train, verbose=0)

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

            time_callback = TimeHistory(
                monitor="acc", min_delta=0.005, patience=2, verbose=0, mode="auto"
            )
            scaler = StandardScaler()
            scaler = scaler.fit(X[train_index])
            X_train = scaler.transform(X[train_index])
            X_test = scaler.transform(X[test_index])
            y_train, y_test = y_bin[train_index], y_bin[test_index]

            history = self._model.fit(
                X_train, y_train, epochs=20, verbose=0, callbacks=[time_callback]
            )

            fit_times.append(time_callback.time)
            train_accuracy.append(history.history["acc"][-1])
            train_precision.append(history.history["precision"][-1])

            result = self._model.evaluate(X_test, y_test, verbose=0)

            test_accuracy.append(result[1])
            test_precision.append(result[2])

        results = {
            "fit_time": np.mean(fit_times).round(decimals=4),
            "train_accuracy": np.mean(train_accuracy).round(decimals=4),
            # "train_precision": np.mean(train_precision).round(decimals=4),
            "test_accuracy": np.mean(test_accuracy).round(decimals=4),
            # "test_precision": np.mean(test_precision).round(decimals=4),
        }

        return {self.name(): results}

    def name(self):
        return "NN-{}-{}-{}-{}".format(
            self._layers,
            self._act_hidden_layers,
            self._act_output_layer,
            self._name_loss,
        )


def create_name(layers, act_hidden, act_output_loss):
    return "NN-{}-{}-{}-{}".format(layers, act_hidden, *act_output_loss)


"""https://stackoverflow.com/a/43186440/3394494"""


class TimeHistory(EarlyStopping):
    def on_train_begin(self, logs={}):
        super().on_train_begin(logs)
        self.time = time.time()

    def on_train_end(self, logs={}):
        super().on_train_end(logs)
        self.time = time.time() - self.time
