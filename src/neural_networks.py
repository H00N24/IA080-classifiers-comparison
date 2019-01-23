from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD

def create_network(n_inputs, n_outputs, params):

    model = Sequential()

    print(n_inputs, n_outputs)

    model.add(Dense(units=500, activation='tanh', input_dim=n_inputs))
    model.add(Dense(units=500, activation='tanh'))
    model.add(Dense(units=2000, activation='tanh'))
    model.add(Dense(units=n_outputs, activation='softmax'))

    model.compile(loss='mean_squared_error',
                  optimizer='sgd')

    return model
