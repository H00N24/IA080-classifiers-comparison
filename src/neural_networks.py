from keras.models import Sequential
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

class KerasWrapper:

    def __init__(self, n_inputs, n_outputs, activation='tanh'):

        self._model = Sequential()

        self._model.add(Dense(units=500, activation=activation, input_dim=n_inputs))
        self._model.add(Dense(units=500, activation=activation))
        self._model.add(Dense(units=2000, activation=activation))
        self._model.add(Dense(units=n_outputs, activation='softmax'))

        self._model.compile(loss='mean_squared_error',
                            optimizer='sgd',
                            metrics=['mae', 'acc'])

    def fit(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        res = self._model.evaluate(X_test, y_test)
        return res

    def print():
        pass
