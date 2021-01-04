import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

class TrainingTracker(keras.callbacks.Callback):
    def __init__(self, X, info, skip_first=False):
        super(TrainingTracker, self).__init__()
        self.data = X.astype(float)
        self.info = info
        self.info["activations"] = []
        self.info["max"] = []
        self.info["min"] = []
        self.skip_first = skip_first

    def on_epoch_end(self, epoch, logs=None):
        num_layers = len(self.model.layers)-(1 if self.skip_first else 0)
        
        if epoch==0:
            self.info["global_max"] = -np.inf
            self.info["global_min"] = np.inf
            self.info["layer_max"]  = [-np.inf]*num_layers
            self.info["layer_min"]  = [np.inf]*num_layers

        mis, mxs = [], []
        A = []
        for i,l in enumerate(self.model.layers[1 if self.skip_first else 0:]):
            lA = K.function([self.model.inputs], [l.output])(self.data)[0]
            mis.append(np.min(lA))
            mxs.append(np.max(lA))
            A.append(lA)
        self.info["activations"].append(A)
        self.info["min"].append(mis)
        self.info["max"].append(mxs)
