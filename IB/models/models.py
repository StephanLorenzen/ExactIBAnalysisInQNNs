import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmo
from tensorflow import keras 
from tensorflow.keras import layers as L

from ..util import quantization as IBQ

def _get_init(init):
    if init not in [None, 'truncated_normal']:
        raise Exception('Unknown initializer...')
    if init=='truncated_normal':
        return lambda l: keras.initializers.TruncatedNormal(mean=0., stddev=1./float(np.sqrt(l)))
    else:
        return lambda l: None

def NN(layers, init=None, quantize=False, fixed_quant=False, num_bits=8):
    # Quantized version
    if quantize and num_bits not in (4,8):
        raise Exception("Unsupported number of bits for quantization")
    
    def _make_layer(l_info, init=None, quantize=False, num_bits=8):
        layer = l_info[0]

        k_init = _get_init(init)
        if layer not in ("Input","Dense","Flatten","Conv2D","MaxPool2D"):
            raise Exception("Unknown layer...")
        
        
        res = None
        act = None
        if layer=="Input":
            # l_info = ("Input", shape)
            res = L.InputLayer(input_shape=l_info[1])
        elif layer=="Dense":
            # l_info = ("Dense", shape, act)
            width, act = l_info[1:]
            uact = "linear" if quantize and act=="tanh" else act
            res = L.Dense(width, activation=uact, kernel_initializer=k_init(width))
        elif layer=="Conv2D":
            # l_info = ("Conv2D", shape, act, kernel)
            width, act, kernel = l_info[1:]
            uact = "linear" if quantize and act=="tanh" else act
            res = L.Conv2D(width, kernel, activation=uact, kernel_initializer=k_init(width))
        elif layer=="MaxPool2D":
            # l_info = ("MaxPool2D", kernel)
            kernel = l_info[1]
            res = L.MaxPooling2D(kernel)
        elif layer=="Flatten":
            res = L.Flatten()
        
        if quantize and layer!="Input":
            q_layer = tfmo.quantization.keras.quantize_annotate_layer
            QC = lambda: None
            if layer not in ("MaxPool2D","Flatten"):
                QC = {
                    4: IBQ.Default4BitConfig,
                    8: IBQ.Default8BitConfig,
                }[num_bits]
                if act == "tanh":
                    QC = {
                        4: IBQ.Tanh4BitConfig,
                        8: IBQ.Tanh8BitConfig,
                    }[num_bits]
            res = q_layer(res, QC())
        return res

    k_layers = []
    for l_info in layers:
        k_layers.append(_make_layer(l_info, init=init, quantize=quantize, num_bits=num_bits))
    model = keras.Sequential(k_layers)

    if quantize:
        # Relevant functions
        q_apply = tfmo.quantization.keras.quantize_apply
        q_scope = tfmo.quantization.keras.quantize_scope

        with q_scope({
            'Default4BitConfig':  IBQ.Default4BitConfig,
            'Default8BitConfig':  IBQ.Default8BitConfig,
            'Tanh4BitConfig':     IBQ.Tanh4BitConfig,
            'Tanh8BitConfig':     IBQ.Tanh8BitConfig,
        }):
            model = q_apply(model)
        
    return model

def FNN(layers, activation='tanh', init=None, quantize=False, fixed_quant=False, num_bits=8):
    k_init = _get_init(init)
    k_layers = [('Input', layers[0])]
    k_layers += [('Dense', l, activation) for l in layers[1:-1]]
    k_layers.append(('Dense', layers[-1], 'softmax'))
    return NN(k_layers, quantize=quantize, fixed_quant=fixed_quant, num_bits=num_bits)

def ShwartzZiv99(activation='tanh', init='truncated_normal', quantize=False, fixed_quant=False, num_bits=8):
    return FNN([12,10,7,5,4,3,2], activation=activation, init=init, quantize=quantize, fixed_quant=fixed_quant, num_bits=num_bits)

def MNIST(init='truncated_normal', quantize=False, fixed_quant=False, num_bits=8):
    k_init = _get_init(init)
    k_layers = []
    k_layers.append(('Input', (28,28,1)))
    k_layers.append(("Conv2D", 32, 'relu', (3,3)))
    k_layers.append(("MaxPool2D", (2,2)))
    k_layers.append(("Conv2D", 64, 'relu', (3,3)))
    k_layers.append(("MaxPool2D", (2,2)))
    k_layers.append(("Conv2D", 64, 'relu', (3,3)))
    k_layers.append(("Flatten",))
    k_layers.append(("Dense", 64, 'relu'))
    k_layers.append(("Dense", 10, 'softmax'))
    return NN(k_layers, quantize=quantize, fixed_quant=fixed_quant, num_bits=num_bits)
