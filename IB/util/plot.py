# Functions for creating various plots related to Deep Learning and the Theory of Deep Learning
# Based on matplotlib

import numpy as np
import matplotlib.pyplot as plt

from . import io as IBio

def information_plane(epochs=None,repeats=None, path=None, est=None):
    if repeats is None:
        if path is None or est is None:
            raise Exception("Missing path or estimator...")
        epochs,repeats,_ = IBio.load_MI(path, est=est)
    else:
        assert epochs is not None
    assert len(epochs)==len(repeats)
    x,y,c = [],[],[]
    for epoch, MIs in zip(epochs,repeats):
        c += [epoch/num_epochs]*len(MIs)
        for (XT,TY) in MIs:
            x.append(XT)
            y.append(TY)
    plt.scatter(x,y,c=c,cmap='inferno')
    plt.show()
    
def mi(var, repeats=None, path=None, est=None):
    if repeats is None:
        if path is None or est is None:
            raise Exception("Missing path or estimator...")
        repeats,stds = IBio.load_MI(path, est=est)
    num_epochs = len(repeats)
    num_layers = len(repeats[0])
    x,ys = list(range(num_epochs)),[[] for _ in range(num_layers)]
    for MIs in repeats:
        for i,(XT,TY) in enumerate(MIs):
            ys[i].append(TY if var.lower()=='y' else XT)
    for i,y in enumerate(ys):
        plt.plot(x,y,label="Layer "+str(i+1))
    plt.legend()
    plt.show()

def accuracy(path=None):
    if path is None:
        raise Exception("Missing path...")
    train, test = IBio.load_accuracy(path)
    train,_,_ = train
    test,_,_ = test
    num_epochs = len(train)
    assert(len(test)==num_epochs)
    epochs = list(range(num_epochs))

    plt.plot(epochs, train, color="darkorange", lw=1)
    plt.plot(epochs, test, color="darkgreen", lw=1)

    plt.show()

def activations(path=None):
    if path is None:
        raise Exception("Missing path...")
    maxs, mins = IBio.load_activations(path)
    num_epochs, num_layers = mins[0].shape
    epochs = list(range(num_epochs))

    colors = ["darkorange","darkgreen","red","blue","yellow","black"]
    for layer in range(num_layers):
        col = colors[layer]
        lmin = mins[0][:,layer]
        lmax = maxs[0][:,layer]
        plt.plot(epochs, lmax, color=col, lw=1, label="max(layer_"+str(layer+1)+")")
        plt.plot(epochs, lmin, '--', color=col, lw=1, label="min(layer_"+str(layer+1)+")")

    plt.legend()
    plt.show()


def activations_2D(path=None):
    assert path!=None, "Path is missing..."
    dfs = IBio.load_activations_2D(path)
    colors = ["darkorange","darkgreen","red","blue","yellow","black","purple","green","pink","orange"]
    
    epoch,df = dfs[-1]
    for i,c in enumerate(colors):
        df2 = df[df["y"]==i]
        plt.scatter(df2["n1"].values,df2["n2"].values,c=c,s=3,label=str(i))
    plt.legend()
    plt.show()

