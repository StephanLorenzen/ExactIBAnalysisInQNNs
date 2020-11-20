# Functions for creating various plots related to Deep Learning and the Theory of Deep Learning
# Based on matplotlib

import numpy as np
import matplotlib.pyplot as plt

from . import io as IBio

def show_image_sample(s):
    plt.figure(figsize=s.shape[:2])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(s)
    plt.show()


def information_plane(repeats=None, path=None, est=None):
    if repeats is None:
        if path is None or est is None:
            raise Exception("Missing path or estimator...")
        repeats,_ = IBio.load_MI(path, est=est)
    num_epochs = len(repeats)
    x,y,c = [],[],[]
    for epoch_idx, MIs in enumerate(repeats):
        c += [epoch_idx/num_epochs]*len(MIs)
        for (XT,TY) in MIs:
            x.append(XT)
            y.append(TY)
    plt.scatter(x,y,c=c,cmap='inferno')
    plt.show()
    
def mi(var, repeats=None, path=None):
    if repeats is None:
        if path is None:
            raise Exception("Missing path...")
        repeats,stds = IBio.load_MI(path)
    num_epochs = len(repeats)
    num_layers = len(repeats[0])
    x,ys = list(range(num_epochs)),[[] for _ in range(num_layers)]
    for MIs in repeats:
        for i,(XT,TY) in enumerate(MIs):
            ys[i].append(TY if var.lower()=='y' else XT)
    for y in ys:
        plt.plot(x,y)
    plt.show()
