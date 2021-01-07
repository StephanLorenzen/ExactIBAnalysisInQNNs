import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

class TrainingTracker(keras.callbacks.Callback):
    def __init__(self, X, info, quantized=False):
        super(TrainingTracker, self).__init__()
        self.data = X.astype(float)
        self.info = info
        self.info["activations"] = []
        self.info["max"] = []
        self.info["min"] = []
        if quantized:
            self.info["unique"] = []
        self.quantized = quantized

    def on_epoch_end(self, epoch, logs=None):
        skip_first = 1 if self.quantized else 0
        num_layers = len(self.model.layers)-skip_first
        
        # Track number of unique values if quantized
        unique = [set() for _ in range(num_layers)]
        
        mis, mxs, unqs = [], [], []
        A = []
        for i,l in enumerate(self.model.layers[skip_first:]):
            lA = K.function([self.model.inputs], [l.output])(self.data)[0]
            mis.append(np.min(lA))
            mxs.append(np.max(lA))
            A.append(lA)
            
            if self.quantized:
                # Track number of unique values if quantized
                unique[i] |= set(np.unique(lA))
                unqs.append(len(unique[i]))
       
        self.info["activations"].append(A)
        self.info["min"].append(mis)
        self.info["max"].append(mxs)
        
        if self.quantized:
            hidden = set()
            for u in unique[:-1]:
                hidden |= u
            unqs.append(len(hidden))
            self.info["unique"].append(unqs)

