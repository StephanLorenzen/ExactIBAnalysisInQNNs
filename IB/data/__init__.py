from scipy import io
from sklearn.model_selection import train_test_split
import numpy as np
import os

def load(dataset_name):
    if dataset_name not in {'tishby'}:
        raise Exception("Unknown data set: '"+dataset_name+"'")
    dataset = {
        'tishby':_load_tishby_data
    }[dataset_name]
    return dataset()

def load_from_path(dataset_path):
    raise Exception("Not implemented!")

# Load data from Tishby paper
def _load_tishby_data():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    # Load data as is
    data = io.loadmat(os.path.join(__location__, 'var_u.mat')) # OBS loads in a weird JSON
    X = data["F"] # (4096, 12)
    y = data["y"] # (1, 4096)
    y = y.squeeze()
    
    return X,y

# Returns (X_train, X_test, y_train, y_test)
def split(X,y,test_frac,seed=None):
    return train_test_split(X, y, random_state=seed, test_size=test_frac, shuffle=True, stratify=y)
