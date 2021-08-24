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
        if self.estimators is None:
            A = []
        else:
            MIest = [[] for est in self.estimators]
        for i,l in enumerate(self.model.layers[skip_first:]):
            lA = K.function([self.model.inputs], [l.output])(self.data)[0]
            mis.append(np.min(lA))
            mxs.append(np.max(lA))
            if self.estimators is None:
                A.append(lA)
            else:
                for j,est in enumerate(self.estimators):
                    MIest[j].append(est([lA])[0])

        if self.estimators is None:
            self.info["activations"].append(A)
        else:
            for i,MI in enumerate(MIest):
                self.info["MI"][i].append(MI)
        self.info["min"].append(mis)
        self.info["max"].append(mxs)
         
