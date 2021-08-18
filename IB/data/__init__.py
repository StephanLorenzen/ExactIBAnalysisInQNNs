from scipy import io
from sklearn.model_selection import train_test_split
import numpy as np
import os

def load(dataset_name):
    if dataset_name not in {'SYN','MNIST','CIFAR'}:
        raise Exception("Unknown data set: '"+dataset_name+"'")
    dataset = {
        'SYN':_load_tishby_data,
        'MNIST':_load_MNIST
    }[dataset_name]
    return dataset()
def load_split(dataset_name):
    if dataset_name not in {'MNIST','CIFAR'}:
        raise Exception("Unknown data set: '"+dataset_name+"'")
    dataset = {
        'MNIST':_load_MNIST_split
    }[dataset_name]
    return dataset()

# Returns (X_train, X_test, y_train, y_test)
def split(X,y,test_frac,seed=None):
    return train_test_split(X, y, random_state=seed, test_size=test_frac, shuffle=True, stratify=y)

# Load data from Tishby paper
def _load_tishby_data():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    # Load data as is
    data = io.loadmat(os.path.join(__location__, 'var_u.mat')) # OBS loads in a weird JSON
    X = data["F"] # (4096, 12)
    y = data["y"] # (1, 4096)
    y = y.squeeze()
    
    return X,y

def _load_MNIST():
    X1,y1,X2,y2 = _load_MNIST_split()
    return np.concatenate((X1,X2),axis=0), np.concatenate((y1,y2),axis=0)
def _load_MNIST_split():
    path = "data/mnist/mnist.data"
    X_train, y_train = _read_idx_file(path, 28*28)
    X_test, y_test   = _read_idx_file(path+".t", 28*28)
    X_train, X_test  = X_train.reshape((-1,28,28,1)), X_test.reshape((-1,28,28,1))
    return X_train, X_test, y_train, y_test
 

def _read_idx_file(path, d, sep=None):
    X = []
    Y = []
    with open(path) as f:
        for l in f:
            x = np.zeros(d)
            l = l.strip().split() if sep is None else l.strip().split(sep)
            Y.append(int(l[0]))
            for pair in l[1:]:
                pair = pair.strip()
                if pair=='':
                    continue
                (i,v) = pair.split(":")
                if v=='':
                    import pdb; pdb.set_trace()
                x[int(i)-1] = float(v)
            X.append(x)
    return np.array(X),np.array(Y)
