import numpy as np
import tensorflow as tf
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

def adaptive(values, n_bins, upper=None, lower=None):
    valflat = values.flatten()
    valflat.sort()
    bsize  = int(len(valflat)/n_bins)
    excess = len(valflat) % n_bins
    bedges = np.cumsum([bsize+1 if i<excess else bsize for i in range(n_bins)])
    bins   = [(valflat[bedge-1]+valflat[bedge])/2 for bedge in bedges[:-1]]
    bins   = [valflat[0]-1] + bins + [valflat[-1]+1]
    return np.digitize(values, bins)

def quantized(values, n_bins=None, upper=None, lower=None):
    upper = np.max(values)
    lower = np.min(values)
    qvals = tf.quantization.quantize(values, lower, upper, tf.qint8)[0]
    # qvals now int8s in range(-128,127). Add 128 to get bins (will convert to int16) 
    return qvals.numpy()+128
