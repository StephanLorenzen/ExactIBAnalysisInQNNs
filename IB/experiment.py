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
        quantize=False,
        data="tishby",
        MI_estimator="binning_uniform",
        repeats=10,
        out_path=None
        ):
    
    # Load data
    print("Loading and preparing data...")
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
    if type(MI_estimator)==list:
        MI_estimators = MI_estimator
    elif type(MI_estimator)==tuple and len(MI_estimator)==3:
        MI_estimators = [MI_estimator]
    else:
        eparams = dict()
        est_func = { 
            "binning_uniform":   IT.estimator.binning_uniform,
            "binning_quantized": IT.estimator.binning_quantized,
            "binning_adaptive":  IT.estimator.binning_adaptive,
            "knn":               IT.estimator.knn,
        }.get(MI_estimator, None)
        if MI_estimator is None:
            raise Exception("Unknown MI estimator!")
        MI_esitmators = [(MI_estimator, est_func, eparams)]

    print("Preparing output dir...")
    # Prepare output directory
    out_path = "out/"+str(int(time())) if out_path is None else out_path
    MI_path  = out_path+"mi/"
    acc_path = out_path+"accuracy/"
    act_path = out_path+"activations/"
    if not os.path.isdir(MI_path):
        os.makedirs(MI_path)
    if not os.path.isdir(acc_path):
        os.makedirs(acc_path)
    if not os.path.isdir(act_path):
        os.makedirs(act_path)
    def _zp(val):
        val = str(val)
        return "0"*(3-len(val)) + val
    
    # Run experiment loop
    print("Starting experiment loop...")
    for it in range(repeats):
        # Train and get activations
        print("> Iteration: "+_zp(it+1))
        ts = time()
        info,train_acc,test_acc = train_model(
                                            model, 
                                            lr, 
                                            epochs, 
                                            (X_train,y_train), 
                                            (X_test,y_test), 
                                            X,
                                            quantize
                                            )
        print(">> Fitting done, elapsed: "+str(int(time()-ts))+"s")
        
        # Compute MI
        print(">> Computing mutual information ("+str(len(MI_estimators))+" estimators)")
        all_activations = []
        for A in info["activations"]:
            all_activations.append(np.concatenate(list(map(lambda x: x.flatten(), A))))
        all_activations = np.concatenate(all_activations)
        
        for (ename, MI_estimator, eparams) in MI_estimators:
            if not os.path.isdir(MI_path+ename+"/"):
                os.makedirs(MI_path+ename+"/")
    
            # Set upper and lower
            if ename=="binning_uniform_"+str(eparams.get("n_bins","")):
                eparams["bins"] = binning.uniform_bins(
                                        all_activations,
                                        eparams["n_bins"],
                                        upper=np.max(info["max"]),
                                        lower=np.min(info["min"])
                                    )
            elif ename=="binning_adaptive_"+str(eparams.get("n_bins","")):
                eparams["bins"] = binning.adaptive_bins(
                                        all_activations,
                                        eparams["n_bins"],
                                        upper=np.max(info["max"]),
                                        lower=np.min(info["min"])
                                    )
            ts = time()
            inls = [(A, y, eparams) for A in info["activations"]]
            #MIs = [MI_estimator(inp) for inp in inls]
            MIs = Pool(16).map(MI_estimator, inls)
            print(">>> Mutual information ("+ename+"), elapsed: "+str(int(time()-ts))+"s")
        
            # Store
            MIs = np.array(MIs)
            np.savetxt(MI_path+ename+"/"+_zp(it+1)+".txt", MIs.reshape(MIs.shape[0],-1))
        
        # Store train and test accuracy and activation min/max
        print(">> Storing training and test accuracies.")
        pd.DataFrame({
            "train_acc":train_acc,
            "test_acc":test_acc
        }).to_csv(acc_path+_zp(it+1)+".csv", index_label="epoch")
        print(">> Storing activation info.")
        col_layers = ["layer_"+str(i+1) for i in range(info["min"].shape[1])]
        pd.DataFrame(
            np.array(info["min"]),
            columns=col_layers
        ).to_csv(act_path+_zp(it+1)+"_min.csv", index_label="epoch")
        pd.DataFrame(
            np.array(info["max"]),
            columns=col_layers
        ).to_csv(act_path+_zp(it+1)+"_max.csv", index_label="epoch")
        pd.DataFrame(
            np.array(info["unique"]),
            columns=col_layers+["hidden_layers"]
        ).to_csv(act_path+_zp(it+1)+"_unique.csv", index_label="epoch")

# Model training
def train_model(model, lr, epochs, train_data, test_data, MI_X, quantize):
    if type(model)==str:
        model = IBmodels.load(model)
    model = model()#quantize=quantize)

    X_train,y_train = train_data
    X_test,y_test   = test_data

    # Start training
    # Output
    info = dict()
    # Options
    loss_fn   = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    callback  = callbacks.TrainingTracker(MI_X, info, quantized=quantize)
    
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
    info["min"] = np.array(info["min"])
    info["max"] = np.array(info["max"])
    info["unique"] = np.array(info["unique"])
    return info, hist.history["accuracy"], hist.history["val_accuracy"]

