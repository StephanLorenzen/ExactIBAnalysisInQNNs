# Functions for creating various plots related to Deep Learning and the Theory of Deep Learning
# Based on matplotlib

import numpy as np
import matplotlib.pyplot as plt

def show_image_sample(s):
    plt.figure(figsize=s.shape[:2])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(s)
    plt.show()

def information_plane(epochs):
    num_epochs, num_layers = epochs.shape
    num_layers //= 2
    x,y,c = [],[],[]
    for epoch_idx, MIs in enumerate(epochs):
        c += [epoch_idx/num_epochs]*num_layers
        for i in range(num_layers):
            #c.append(i)
            x.append(MIs[2*i])
            y.append(MIs[2*i+1])
    import pdb; pdb.set_trace()
    plt.scatter(x,y,c=c,cmap='inferno')
    plt.show()
def old_information_plane(MIs):
    num_epochs = len(MIs)
    num_layers = len(MIs[0])
    x,y,c = [],[],[]
    for idx,ep in enumerate(MIs):
        c += [idx/num_epochs]*num_layers
        x += [e[0] for e in ep]
        y += [e[1] for e in ep]
    
    plt.scatter(x,y,c=c,cmap='inferno')
    plt.show()
    

