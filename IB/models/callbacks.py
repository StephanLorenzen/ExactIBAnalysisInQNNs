import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

class TrainingTracker(keras.callbacks.Callback):
    def __init__(self, X, info, estimators=None, quantized=False):
        super(TrainingTracker, self).__init__()
        self.data = X.astype(float)
        self.info = info
        self.estimators = estimators
        if self.estimators is None:
            self.info["activations"] = []
        else:
            self.info["MI"] = [[] for _ in self.estimators]
        self.info["max"] = []
        self.info["min"] = []
        self.quantized = quantized

    def on_epoch_end(self, epoch, logs=None):
        skip_first = 1 if self.quantized else 0
        num_layers = len(self.model.layers)-skip_first
        
        mis, mxs = [], []
        A = []
        for i,l in enumerate(self.model.layers[skip_first:]):
            lA = K.function([self.model.inputs], [l.output])(self.data)[0]
            mis.append(np.min(lA))
            mxs.append(np.max(lA))
            A.append(lA)
        
        if self.estimators is None:
            self.info["activations"].append(A)
        else:
            for i,est in enumerate(self.estimators):
                self.info["MI"][i].append(est(A))
        self.info["min"].append(mis)
        self.info["max"].append(mxs)
         
