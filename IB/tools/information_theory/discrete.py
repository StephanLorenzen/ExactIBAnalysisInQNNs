import numpy as np

def distribution_from_bins(bins):
    _, counts = np.unique(bins, axis=0, return_counts=True)
    return counts/sum(counts)

def entropy(X):
    dX = distribution_from_bins(X)
    return -np.sum(dX * np.log2(dX))

def conditional_entropy(X,Y):
    ys, ycnt = np.unique(Y, return_counts=True)
    cond_entropy = 0
    for y,cnt in zip(ys,ycnt):
        cond_entropy += entropy(X[Y==y])*(cnt/len(Y))
    return cond_entropy

def mutual_information(X,Y):
    return entropy(X)-conditional_entropy(X,Y)
