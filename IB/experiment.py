import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from time import time

from multiprocessing import Pool

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
        prefit_random=0,
        repeats=1,
        out_path=None,
        start_from=1,
        low_memory=False,
        ):
   
    seed = 1000

    train, test, (X,y) = prep_data(data,seed)
    
    print("Preparing output dir...")
    # Prepare output directory
    out_path = "out/"+str(int(time())) if out_path is None else out_path
    MI_path  = out_path+"mi/"
    acc_path = out_path+"accuracy/"
    act_path = out_path+"activations/"
    _2D_path  = act_path+"2D/"
    for path in [MI_path, acc_path, act_path, _2D_path]:
        if not os.path.isdir(path):
            os.makedirs(path)
    
    # Run experiment loop
    print("Starting experiment loop...")

    lmest = None
    if low_memory:
        lmest = []
        for Est in MI_estimators:
            if Est.require_setup():
                raise Exception("Estimator requires setup - cannot use low memory setting!")
            lmest.append(lambda A: Est(A,y))
    restarts = 0
    rep = start_from-1
    while rep<repeats:
        print(">>> Iteration: "+_zp(rep+1))

        # Train and get activations
        print(">> Fitting model, "+str(epochs)+" epochs")
        ts = time()
        if prefit_random>0: print(">>> Prefitting to random labels: "+str(prefit_random)+" epochs")
        info, train_acc, test_acc = train_model(Model,lr,batch_size,epochs,train,test,X,prefit_random=prefit_random,estimators=lmest, seed=seed+rep+restarts)
        if info is None:
            print("!!! Restarting repetition")
            restarts += 1
            continue
        else:
            rep += 1
            restarts = 0
        print(">> Fitting done, elapsed: "+str(int(time()-ts))+"s")
        print(">> Computing mutual information ("+str(len(MI_estimators))+" estimators)")
        for i,Est in enumerate(MI_estimators):
            path = MI_path+Est.dir()+"/"
            if not os.path.isdir(path):
                os.makedirs(path)
            if low_memory:
                print(">>> Storing MI, "+str(Est))
                MIs_prefit = np.array(info["prefit"]["MI"][i]) if "prefit" in info else None
                MIs = np.array(info["MI"][i])
            else:
                print(">>> Estimating, "+str(Est))
                ts = time()
                MIs_prefit = compute_MI(info["prefit"]["activations"]) if "prefit" in info else None
                MIs = compute_MI(Est, info["activations"], y)
                print(">>> Mutual information computed, elapsed: "+str(int(time()-ts))+"s")
            
            n_epoch = MIs.shape[0]
            MIs = np.concatenate([info["epoch"].reshape(n_epoch,-1),MIs.reshape(n_epoch,-1)], axis=1)
            np.savetxt(path+_zp(rep)+".txt", MIs)
            if MIs_prefit is not None:
                n_prefit = MIs_prefit.shape[0]
                MIs_prefit = np.concatenate([info["prefit"]["epoch"].reshape(n_prefit,-1),MIs_prefit.reshape(n_prefit,-1)], axis=1)
                np.savetxt(path+_zp(rep)+"_prefit.txt", MIs_prefit)

        # Store train and test accuracy and activation min/max
        print(">> Storing training and test accuracies.")
        pd.DataFrame({
            "train_acc":train_acc,
            "test_acc":test_acc
        }).to_csv(acc_path+_zp(rep)+".csv", index_label="epoch")
        
        print(">> Storing activation info.")
        col_layers = ["layer_"+str(i+1) for i in range(info["min"].shape[1])]
        df_min = pd.DataFrame(np.array(info["min"]),columns=col_layers)
        df_min.index = info["epoch"]
        df_min.to_csv(act_path+_zp(rep)+"_min.csv", index_label="epoch")
        df_max = pd.DataFrame(np.array(info["max"]),columns=col_layers)
        df_max.index = info["epoch"]
        df_max.to_csv(act_path+_zp(rep)+"_max.csv", index_label="epoch")

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
    else:
        raise Exception("Unknown data set: '"+data+"'")
    return (X_train, y_train), (X_test, y_test), (X,y)

# Model training
def train_model(Model, lr, batch_size, epochs, train_data, test_data, X, prefit_random=False, estimators=None, seed=None):
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
    compute_freq = 5 if epochs >= 8000 else 2
    
    # Prefit random:
    if prefit_random>0:
        # prefit_random is number of epochs to prefit
        rand_info = dict()
        cb = callbacks.TrainingTracker(
                X,
                rand_info,
                estimators=estimators,
                quantized=quantized,
                compute_MI_freq=compute_freq)
        
        # Shuffle labels
        random_y_train = tf.random.shuffle(y_train)
        model.fit(
            X_train,
            random_y_train,
            batch_size=batch_size,
            epochs=prefit_random,
            callbacks=[cb],
            verbose=0
        )
        rand_info["epoch"] = np.array(rand_info["epoch"])
        rand_info["min"] = np.array(rand_info["min"])
        rand_info["max"] = np.array(rand_info["max"])
        info["prefit"] = rand_info

    # Callback
    callback = callbacks.TrainingTracker(
        X, 
        info, 
        estimators=estimators, 
        quantized=quantized,
        compute_MI_freq=compute_freq,
        auto_stop=True
    )
    # Fit
    hist = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[callback],
            validation_data=test_data,
            verbose=0
            )
    if model.stop_training:
        print("!!! Stopped training - bad fit.")
        return None, None, None
    info["epoch"] = np.array(info["epoch"])
    info["min"] = np.array(info["min"])
    info["max"] = np.array(info["max"])
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

