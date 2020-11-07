import numpy as np
# @param values : [np.array(shape=(n,layer_d))], len = L
# @return       : [], len = L

def uniform(values, n_bins, upper='max', lower='min'):
    if upper=='max':
        upper = np.max(values)
    if lower=='min':
        lower = np.min(values)
    step = (upper-lower)/n_bins
    bins = np.arange(lower, upper+step/2, step)
    return np.digitize(values, bins)

def uniform_quantized(values, n=8):
    # Create 2^n bins, where usually n=8
    ivalues = values.astype(int)
    _,counts = np.unique(values,return_counts=True)
    return counts

def adaptive(values, n_bins):
    pass
