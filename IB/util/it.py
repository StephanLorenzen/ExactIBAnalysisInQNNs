import numpy as np

def distribution_from_bins(bins):
    # bins in N^(n times l)
    _, counts = np.unique(bins, axis=0, return_counts=True)
    # counts in N^n
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

def split_entropy(X):
    return np.apply_along_axis(entropy,0,X).mean()

def split_mutual_information(X,Y):
    return (np.apply_along_axis(entropy,0,X)-np.apply_along_axis(lambda x: conditional_entropy(x,Y),0,X)).mean()
