import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmo
from tensorflow import keras 
from tensorflow.keras import layers as L

from ..util import quantization as IBQ

def FNN(layers, input_dim=12, activation='tanh', init=None, quantize=False, fixed_quant=False, num_bits=8):
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
        if num_bits not in (4,8):
            raise Exception("Unsupported number of bits for quantization")

        # Relevant functions
        q_layer = tfmo.quantization.keras.quantize_annotate_layer
        q_apply = tfmo.quantization.keras.quantize_apply
        q_scope = tfmo.quantization.keras.quantize_scope

        # Relevant QuantizeConfigs
        DQC = {
            4: IBQ.Default4BitConfig,
            8: IBQ.Default8BitConfig,
        }[num_bits]
        QConfig = DQC
        if activation == "tanh":
            activation = "linear"
            QConfig = {
                4: IBQ.Tanh4BitConfig,
                8: IBQ.Tanh8BitConfig,
            }[num_bits]
            #QConfig = IBQ.TanhFixedQuantizeConfig if fixed_quant else IBQ.TanhQuantizeConfig
        k_layers = [keras.layers.InputLayer(input_shape=input_dim)]
        for l in layers:
            qconf = QConfig()#num_bits=num_bits) #None if QConfig is None else QConfig()
            k_layers.append(
                q_layer(L.Dense(l, activation=activation, kernel_initializer=k_init(l)), qconf)
            )
        k_layers.append(q_layer(L.Dense(2, activation='softmax'), DQC()))#IBQ.DefaultDenseQuantizeConfig(num_bits=num_bits)))
        model = keras.Sequential(k_layers)
        with q_scope({
            'Default4BitConfig':  IBQ.Default4BitConfig,
            'Default8BitConfig':  IBQ.Default8BitConfig,
            'Tanh4BitConfig':     IBQ.Tanh4BitConfig,
            'Tanh8BitConfig':     IBQ.Tanh8BitConfig,
            #'DefaultDenseQuantizeConfig' : IBQ.DefaultDenseQuantizeConfig,
            #'TanhQuantizeConfig': IBQ.TanhQuantizeConfig,
            #'TanhFixedQuantizeConfig': IBQ.TanhFixedQuantizeConfig,
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

def shwartz_ziv_99(activation='tanh', init='truncated_normal', quantize=False, fixed_quant=False, num_bits=8):
    return FNN([10,7,5,4,3], input_dim=12, activation=activation, init=init, quantize=quantize, fixed_quant=fixed_quant, num_bits=num_bits)

