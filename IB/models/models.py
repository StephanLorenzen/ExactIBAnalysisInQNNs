import numpy as np
import tensorflow as tf
from tensorflow import keras 

def shwartz_ziv_99(activation='tanh'):
    layers = [10,7,5,4,3] # input dim=12, output neurons (2) added in the end
    model = keras.Sequential()
    for i,lsize in enumerate(layers):
        l_init = keras.initializers.TruncatedNormal(mean=0., stddev=1./float(np.sqrt(lsize)))
        if i==0:
            model.add(keras.layers.Dense(lsize, input_dim=12, activation=activation, kernel_initializer=l_init))
        else:
            model.add(keras.layers.Dense(lsize, activation=activation, kernel_initializer=l_init))
    model.add(keras.layers.Dense(2, activation='softmax'))
    return model
