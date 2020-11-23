import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

class StoreActivations(keras.callbacks.Callback):
    def __init__(self, X, out, skip_first=False):
        super(StoreActivations, self).__init__()
        self.data = X.astype(float)
        self.acts = out
        self.skip_first = skip_first

    def on_epoch_end(self, epoch, logs=None):
        A = []
        for l in self.model.layers[1 if self.skip_first else 0:]:
            A.append(K.function([self.model.inputs], [l.output])(self.data)[0])
        self.acts.append(A)
        
        # Other method, slow
        #_l = [l.output for l in self.model.layers[1:]]
        #_m = keras.Model(inputs=self.model.inputs, outputs=_l)
        #self.acts.append(_m.predict(self.data))
