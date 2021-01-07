import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmo
from tensorflow import keras 
from tensorflow.keras import layers as L

from ..tools import quantization as IBQ

def FNN(layers, input_dim=12, activation='tanh', init=None, quantize=False):
    if init not in [None, 'truncated_normal']:
        raise Exception('Unknown initializer...')
    
    if init=='truncated_normal':
        def k_init(l):
            return keras.initializers.TruncatedNormal(mean=0., stddev=1./float(np.sqrt(l)))
    else:
        def k_init(l):
            return None
    
    if quantize:
        # Quantized version

        # Relevant functions
        q_layer = tfmo.quantization.keras.quantize_annotate_layer
        q_apply = tfmo.quantization.keras.quantize_apply
        q_scope = tfmo.quantization.keras.quantize_scope

        # Relevant QuantizeConfigs
        QConfig = {
            "tanh":IBQ.TanhQuantizeConfig,
            "relu6":IBQ.ReLU6QuantizeConfig,
        }.get(activation, IBQ.DefaultDenseQuantizeConfig)
        k_layers = [keras.layers.InputLayer(input_shape=input_dim)]
        for l in layers:
            k_layers.append(
                q_layer(L.Dense(l, activation="linear", kernel_initializer=k_init(l)), QConfig())
            )
        k_layers.append(q_layer(L.Dense(2, activation='linear'),IBQ.SoftMaxQuantizeConfig()))
        model = keras.Sequential(k_layers)
        
        with q_scope({
            'DefaultDenseQuantizeConfig': IBQ.DefaultDenseQuantizeConfig,
            'ReLU6QuantizeConfig': IBQ.ReLU6QuantizeConfig,
            'TanhQuantizeConfig': IBQ.TanhQuantizeConfig,
            'SoftMaxQuantizeConfig': IBQ.SoftMaxQuantizeConfig,
        }):
            model = q_apply(model)
        
        return model
    else:
        # Standard model

        k_layers = [keras.layers.InputLayer(input_shape=input_dim)]
        for l in layers:
            k_layers.append(L.Dense(l, activation=activation, kernel_initializer=k_init(l)))
        k_layers.append(L.Dense(2, activation='softmax'))

        return keras.Sequential(k_layers)

def shwartz_ziv_99(activation='tanh', init='truncated_normal', quantize=False):
    return FNN([10,7,5,4,3], input_dim=12, activation=activation, init=init, quantize=quantize)

