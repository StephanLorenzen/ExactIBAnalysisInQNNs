import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from time import time

from multiprocessing import Pool

from . import data as IBdata
from .tools import callbacks, binning, it as IT
from . import models as IBmodels

# Default params
_DEFAULT_MODEL        = ("shwartz_ziv_99",  {"lr":10**-4, "activation":"tanh", "epochs":1000})
_DEFAULT_MI_ESTIMATOR = ("binning-uniform", {"n_bins":100,"min":-1.0,"max":1.0})

def run(setup):
    if setup is None or type(setup)!=dict:
        raise Exception("Missing parameters!")
    

    # Load model
    print("Loading model...")
    # Model and model parameters
    model, mparams = setup.get("model", _DEFAULT_MODEL)
    if type(model)==str:
        model = IBmodels.load(model, mparams.get('activation','tanh'))
    if model is None:
        raise Exception("Unknown model!")

    # Load data
    print("Loading data...")
    dataset = setup.get("data","tishby")
    if type(dataset)==str:
        data = IBdata.load(dataset)
        if data is None:
            # Might be path
            data = IBdata.load_from_path(dataset)
            if data is None:
                raise Exception("Unknown data set or path: '"+dataset+"'")
    else:
        data = dataset    
    X,y = data 
    X_train, X_test, y_train, y_test = IBdata.split(X,y,0.2)
    y_train = tf.one_hot(y_train,2)
    y_test  = tf.one_hot(y_test,2)

    # Run experiment loop
    print("Starting experiment loop...")
    num_iterations = setup.get("n_iterations",1)
    mi_estimator_name, mi_params = setup.get("mi_estimator", _DEFAULT_MI_ESTIMATOR)
    mi_estimator   = {
        "binning-uniform":binning_uniform,
        "binning-adaptive":binning_adaptive,
        "knn":knn,
    }.get(mi_estimator_name, None)
    if mi_estimator is None:
        raise Exception("Unknown MI estimator!")
    
    # Prepare output directory
    out_path = setup.get("out_path","out/")
    #print("Saving outputs to '"+out_path+"'...")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    def _zp(val):
        val = str(val)
        return "0"*(3-len(val)) + val
    
    print("\tn_iterations: "+str(num_iterations))
    print("\tmi_estimator: "+str(mi_estimator_name)+", params: "+str(mi_params))
    print("\tmodel parameters: "+str(mparams))
    for it in range(num_iterations):
        # Train and get activations
        print(">Iteration: "+_zp(it))
        ts = time()
        model = IBmodels.load("shwartz_ziv_99", mparams.get('activation','tanh'))
        activations,accs = train_model(model, mparams, (X_train,y_train), (X_test,y_test), X)
        print(">> Fitting done, elapsed: "+str(int(time()-ts))+"s")
        # Compute MI
        ts = time()
        inls = [(A, y, mi_params) for A in activations]
        MIs = Pool(16).map(mi_estimator, inls)
        print(">> Mutual information computed, elapsed: "+str(int(time()-ts))+"s")
        
        # Store
        MIs = np.array(MIs)
        np.savetxt(out_path+_zp(it+1)+"_accuracy.txt", accs)
        np.savetxt(out_path+_zp(it+1)+"_mi.txt", MIs.reshape(MIs.shape[0],-1))
        #output.append((accs,np.array(MIs)))
   
    return
    # Done, store
    for it,(accs, MIs) in enumerate(output):
        np.savetxt(out_path+_zp(it)+"_accuracy.txt", accs)
        np.savetxt(out_path+_zp(it)+"_mi.txt", MIs.reshape(MIs.shape[0],-1))


# Model training
def train_model(model, mparams, train_data, test_data, MI_X):
    # Get model parameters
    lr     = mparams.get('lr', 10**-4)
    epochs = mparams.get('epochs', 2000)
    
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
    model.fit(X_train,y_train,batch_size=len(X_train),epochs=epochs,callbacks=[callback],verbose=0)

    return A,[]


# MI estimators
def binning_uniform(inp):
    A, Y, params = inp
    params = dict() if params is None else params
    n_bins = params.get("n_bins", 30)
    up,lw  = params.get("max","max"), params.get("min","min")
    MI_layers = []
    for layer in A:
        T = binning.uniform(layer,n_bins,lower=lw,upper=up)
        MI_XT = IT.discrete.entropy(T)
        MI_TY = IT.discrete.mutual_information(T,Y)
        MI_layers.append((MI_XT,MI_TY))
    return MI_layers

def binning_adaptive(A, Y, params):
    pass

def knn(A, Y, params):
    pass
