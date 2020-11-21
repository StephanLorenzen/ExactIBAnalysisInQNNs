import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers as L

def FNN(layers, input_dim=12, activation='tanh', init=None):
    if init not in [None, 'truncated_normal']:
        raise Exception('Unknown initializer...')
    
    if init=='truncated_normal':
        def k_init(l):
            return keras.initializers.TruncatedNormal(mean=0., stddev=1./float(np.sqrt(l)))
    else:
        def k_init(l):
            return None

    k_layers = [keras.layers.InputLayer(input_shape=input_dim)]
    for l in layers:
        k_layers.append(L.Dense(l, activation=activation, kernel_initializer=k_init(l)))
    k_layers.append(L.Dense(2, activation='softmax'))

    return keras.Sequential(k_layers)

def shwartz_ziv_99(activation='tanh', init='truncated_normal'):
    return FNN([10,7,5,4,3], input_dim=12, activation=activation, init=init)

def _shwartz_ziv_99(activation='tanh'):
    return keras.Sequential([
        keras.layers.InputLayer(input_shape=12),
        keras.layers.Dense(10, activation=activation),
        keras.layers.Dense(7, activation=activation),
        keras.layers.Dense(5, activation=activation),
        keras.layers.Dense(4, activation=activation),
        keras.layers.Dense(3, activation=activation),
        keras.layers.Dense(2, activation="softmax")
    ])
