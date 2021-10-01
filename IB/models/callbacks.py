import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

class TrainingTracker(keras.callbacks.Callback):
    def __init__(self, X, info, estimators=None, quantized=False, model_save_path=None):
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
        self.model_save_path = model_save_path
        self.save_epochs = set(list(range(1,10+1))+list(range(10,101,10))+list(range(200,8001,200)))

    def on_epoch_end(self, epoch, logs=None):
        skip_first = 1 if self.quantized else 0
        mis, mxs = [], []
        if self.estimators is None:
            A = []
        else:
            MIest = [[] for est in self.estimators]
        for i,l in enumerate(self.model.layers[skip_first:]):
            if "flatten" in l.name:
                continue # Same neurons as previous layer - skip
            if self.estimators is None:
                lA = K.function([self.model.inputs], [l.output])(self.data)[0]
            else:
                lA = []
                for part in np.array_split(self.data, 10):
                    lA.append(K.function([self.model.inputs], [l.output])(part)[0])
                lA = np.concatenate(lA)
            mis.append(np.min(lA))
            mxs.append(np.max(lA))
            if len(lA.shape)==2 and lA.shape[1]==2 and (epoch+1) in self.save_epochs:
                # Two dimensional and flat -> store for plot
                if "acts_2D" not in self.info:
                    self.info["acts_2D"] = []
                self.info["acts_2D"].append((epoch,lA))
                if self.model_save_path!=None:
                    self.model.save(self.model_save_path+str(epoch+1))
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
         
