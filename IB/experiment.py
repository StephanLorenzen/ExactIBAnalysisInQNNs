import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from time import time

from multiprocessing import Pool

from carbontracker.tracker import CarbonTracker as CT

from . import data as IBdata, models as IBmodels
from .models import callbacks
from .util import estimator

def run_experiment(
        Model,
        MI_estimators,
        data,
        lr=10**-4,
        batch_size=256,
        epochs=8000,
        repeats=1,
        out_path=None,
        start_from=1,
        low_memory=False,
        use_carbontracker=False,
        ):
   
    seed = 1000

    train, test, (X,y) = prep_data(data,seed)
    
    print("Preparing output dir...")
    # Prepare output directory
    out_path = "out/"+str(int(time())) if out_path is None else out_path
    MI_path  = out_path+"mi/"
    acc_path = out_path+"accuracy/"
    act_path = out_path+"activations/"
    for path in [MI_path, acc_path, act_path]:
        if not os.path.isdir(path):
            os.makedirs(path)
    
    # Run experiment loop
    print("Starting experiment loop...")
    if use_carbontracker:
        tracker = CT(epochs=1)

    lmest = None
    if low_memory:
        lmest = []
        for Est in MI_estimators:
            if Est.require_setup():
                raise Exception("Estimator requires setup - cannot use low memory setting!")
            lmest.append(lambda A: Est(A,y))
    for rep in range(start_from-1, repeats):
        print(">>> Iteration: "+_zp(rep+1))
        if use_carbontracker:
            tracker.epoch_start()

        # Train and get activations
        print(">> Fitting model, "+str(epochs)+" epochs")
        ts = time()
        info, train_acc, test_acc = train_model(Model,lr,batch_size,epochs,train,test,X,estimators=lmest, seed=seed+rep) 
        print(">> Fitting done, elapsed: "+str(int(time()-ts))+"s")

        print(">> Computing mutual information ("+str(len(MI_estimators))+" estimators)")
        for i,Est in enumerate(MI_estimators):
            path = MI_path+Est.dir()+"/"
            if not os.path.isdir(path):
                os.makedirs(path)
            if low_memory:
                print(">>> Storing MI, "+str(Est))
                MIs = np.array(info["MI"][i])
            else:
                print(">>> Estimating, "+str(Est))
                ts = time()
                MIs = compute_MI(Est, info["activations"], y)
                print(">>> Mutual information computed, elapsed: "+str(int(time()-ts))+"s")
            np.savetxt(path+_zp(rep+1)+".txt", MIs.reshape(MIs.shape[0],-1))

        # Store train and test accuracy and activation min/max
        print(">> Storing training and test accuracies.")
        pd.DataFrame({
            "train_acc":train_acc,
            "test_acc":test_acc
        }).to_csv(acc_path+_zp(rep+1)+".csv", index_label="epoch")
        print(">> Storing activation info.")
        col_layers = ["layer_"+str(i+1) for i in range(info["min"].shape[1])]
        pd.DataFrame(
            np.array(info["min"]),
            columns=col_layers
        ).to_csv(act_path+_zp(rep+1)+"_min.csv", index_label="epoch")
        pd.DataFrame(
            np.array(info["max"]),
            columns=col_layers
        ).to_csv(act_path+_zp(rep+1)+"_max.csv", index_label="epoch")
        if use_carbontracker:
            tracker.epoch_stop()
    if use_carbontracker:
        tracker.stop()

def _zp(val):
    val = str(val)
    return "0"*(3-len(val)) + val

def prep_data(data, seed):
    # Load data
    print("Loading and preparing data...")
    if data=="SYN":
        X,y = IBdata.load(data)
        X_train, X_test, y_train, y_test = IBdata.split(X,y,0.2,seed=seed)
        y_train = tf.one_hot(y_train,2)
        y_test  = tf.one_hot(y_test,2)
    elif data in ("MNIST","CIFAR"):
        X_train, X_test, y_train, y_test = IBdata.load_split(data)
        X,y = np.concatenate((X_train,X_test),axis=0), np.concatenate((y_train,y_test),axis=0)
        y_train = tf.one_hot(y_train,10)
        y_test  = tf.one_hot(y_test,10)
    return (X_train, y_train), (X_test, y_test), (X,y)

# Model training
def train_model(Model, lr, batch_size, epochs, train_data, test_data, X, estimators=None, seed=None):
    model, quantized = Model()
    
    X_train,y_train = train_data
    X_test,y_test   = test_data

    if batch_size is None:
        batch_size = len(X_train)

    if seed is not None:
        tf.random.set_seed(seed)

    # Start training
    loss_fn   = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,loss=loss_fn,metrics=['accuracy'])
        
    # Output
    info = dict()
   
    # Callback
    callback = callbacks.TrainingTracker(X, info, estimators=estimators, quantized=quantized)
    hist = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[callback],
            validation_data=test_data,
            verbose=0
            )
    info["min"] = np.array(info["min"])
    info["max"] = np.array(info["max"])
    if "unique" in info:
        info["unique"] = np.array(info["unique"])
    return info, hist.history["accuracy"], hist.history["val_accuracy"]
def _apply_estimator(inp):
    A, y, Est = inp
    return Est(A,y)

def compute_MI(Estimator, activations, y):
    # Estimator setup
    Estimator.setup(activations)
    # Prepare input
    inps = [(A, y, Estimator) for A in activations]
    # Process
    MIs = Pool(16).map(_apply_estimator, inps)
    return np.array(MIs)

