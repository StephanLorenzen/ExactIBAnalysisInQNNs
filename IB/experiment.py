import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from time import time

from multiprocessing import Pool

from . import data as IBdata
from .tools import callbacks, binning, information_theory as IT
from . import models as IBmodels

"""
model : a tensorflow model or a name in {shwartz_ziv_99}

"""
def run_experiment(
        model="shwartz_ziv_99",
        lr=10**-4,
        epochs=10,
        data="tishby",
        MI_estimator="binning_uniform",
        repeats=10,
        out_path=None
        ):
    
    # Load data
    print("Loading and preparing data...")
    #dataset = setup.get("data","tishby")
    _data = data
    if type(data)==str:
        _data = IBdata.load(data)
        if _data is None:
            # Might be path
            _data = IBdata.load_from_path(data)
            if data is None:
                raise Exception("Unknown data set or path: '"+data+"'")
    X,y = _data 
    X_train, X_test, y_train, y_test = IBdata.split(X,y,0.2)
    y_train = tf.one_hot(y_train,2)
    y_test  = tf.one_hot(y_test,2)

    print("Preparing MI estimator...")
    if type(MI_estimator)==tuple and len(MI_estimator)==2:
        MI_estimator, eparams = MI_estimator
    else:
        eparams = dict()
        MI_estimator = { 
            "binning_uniform":   IT.estimator.binning_uniform,
            "binning_quantized": IT.estimator.binning_quantized,
            "binning_adaptive":  IT.estimator.binning_adaptive,
            "knn":               IT.estimator.knn,
        }.get(MI_estimator, None)
    if MI_estimator is None:
        raise Exception("Unknown MI estimator!")
    
    print("Preparing output dir...")
    # Prepare output directory
    out_path = "out/"+str(int(time())) if out_path is None else out_path
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    def _zp(val):
        val = str(val)
        return "0"*(3-len(val)) + val
    
    # Run experiment loop
    print("Starting experiment loop...")
    for it in range(repeats):
        # Train and get activations
        print("> Iteration: "+_zp(it+1))
        ts = time()
        _model = IBmodels.load(model)() if type(model)==str else model()
        activations,train_acc,test_acc = train_model(
                                                    _model, 
                                                    lr, 
                                                    epochs, 
                                                    (X_train,y_train), 
                                                    (X_test,y_test), 
                                                    X
                                                    )
        print(">> Fitting done, elapsed: "+str(int(time()-ts))+"s")
        
        # Compute MI
        ts = time()
        inls = [(A, y, eparams) for A in activations]
        MIs = Pool(16).map(MI_estimator, inls)
        print(">> Mutual information computed, elapsed: "+str(int(time()-ts))+"s")
        
        # Store
        MIs = np.array(MIs)
        pd.DataFrame({
            "train_acc":train_acc,
            "test_acc":test_acc
        }).to_csv(out_path+_zp(it+1)+"_accuracy.txt", index_label="epoch")
        np.savetxt(out_path+_zp(it+1)+"_mi.txt", MIs.reshape(MIs.shape[0],-1))


# Model training
def train_model(model, lr, epochs, train_data, test_data, MI_X):
    X_train,y_train = train_data
    X_test,y_test   = test_data

    # Start training
    # Output
    A = []
    # Options
    loss_fn   = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    callback  = callbacks.StoreActivations(MI_X, A)
    
    model.compile(optimizer=optimizer,loss=loss_fn,metrics=['accuracy'])
    hist = model.fit(
            X_train,
            y_train,
            batch_size=len(X_train),
            epochs=epochs,
            callbacks=[callback],
            validation_data=test_data,
            verbose=0
            )
    return A, hist.history["accuracy"], hist.history["val_accuracy"]

