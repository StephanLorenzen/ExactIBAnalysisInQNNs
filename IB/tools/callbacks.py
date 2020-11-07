import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from . import binning, it#, derr

class StoreActivations(keras.callbacks.Callback):
    def __init__(self, X, out):
        super(StoreActivations, self).__init__()
        self.data = X
        self.acts = out

    def on_epoch_end(self, epoch, logs=None):
        A = []
        for l in self.model.layers:
            A.append(K.function([self.model.layers[0].input], [l.output])(self.data)[0])

        self.acts.append(A)

class ComputeMI(keras.callbacks.Callback):
    def __init__(self, X, y, out, params):
        super(ComputeMI, self).__init__()
        self.data = (X,y)
        self.MIs = out
        self.n_bins = params['num_bins']
        self.b_up   = params['up']
        self.b_lw   = params['lw']

    def on_epoch_end(self, epoch, logs=None):
        inp  = self.model.input
        outs = [l.output for l in self.model.layers]
        funcs = [K.function([inp], [out]) for out in outs]
        A = [func([self.data[0], 1.]) for func in funcs]
        # A list of L layers, each layer l of size m is a list of n x m activations
        import pdb; pdb.set_trace() 
        MI_epoch = []
        y = self.data[1]
        for layer in A:
            layer = layer[0]
            T = binning.uniform(layer,self.n_bins,lower=self.b_lw,upper=self.b_up)
            MI_XT = it.entropy(T) #it.mutual_information(X,T)
            MI_TY = it.mutual_information(T,y)
            MI_epoch.append((MI_XT,MI_TY))
        self.MIs.append(MI_epoch)
