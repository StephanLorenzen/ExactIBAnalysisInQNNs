import numpy as np
import tensorflow as tf
# @param values : [np.array(shape=(n,layer_d))], len = L
# @return       : [], len = L

def uniform_bins(n_bins, values=None, upper='max', lower='min'):
    assert(values is not None or (upper!='max' and lower!='min'))
    if upper=='max':
        upper = np.max(values)
    if lower=='min':
        lower = np.min(values)
    if upper==lower:
        upper = lower+10**-9
    step = (upper-lower)/n_bins
    bins = np.arange(lower, upper, step)
    return bins

def uniform(n_bins, values=None, upper='max', lower='min'):
    bins = uniform_bins(n_bins, values, upper, lower)
    return np.digitize(values, bins)

def adaptive_bins(n_bins, values):
    valflat = values.flatten()
    valflat.sort()
    bsize  = int(len(valflat)/n_bins)
    excess = len(valflat) % n_bins
    bedges = np.cumsum([bsize+1 if i<excess else bsize for i in range(n_bins)])
    bins   = [(valflat[bedge-1]+valflat[bedge])/2 for bedge in bedges[:-1]]
    bins   = [valflat[0]-1] + bins + [valflat[-1]+1]
    return bins

def adaptive(n_bins, values):
    bins = adaptive_bins(n_bins, values)
    return np.digitize(values, bins)

def quantized(values):
    upper = np.max(values)
    lower = np.min(values)
    qvals = tf.quantization.quantize(values, lower, upper, tf.qint8)[0]
    # qvals now int8s in range(-128,127). Add 128 to get bins (will convert to int16) 
    return qvals.numpy()+128
