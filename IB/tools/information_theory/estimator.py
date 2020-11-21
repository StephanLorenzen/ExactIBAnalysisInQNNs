from .. import binning
from . import discrete

# MI estimators
def binning_uniform(inp):
    return _binning(inp, binning.uniform)

def binning_adaptive(inp):
    return _binning(inp, binning.adaptive)

def binning_quantized(inp):
    return _binning(inp, binning.quantized)

def _binning(inp, bin_func):
    A, Y, params = inp
    params = dict() if params is None else params
    n_bins = params.get("n_bins", 30)
    up,lw  = params.get("upper","max"), params.get("lower","min")
    lw = "min" if lw is None else lw
    up = "max" if up is None else up
    MI_layers = []
    for layer in A:
        T = bin_func(layer,n_bins,lower=lw,upper=up)
        MI_XT = discrete.entropy(T)
        MI_TY = discrete.mutual_information(T,Y)
        MI_layers.append((MI_XT,MI_TY))
    return MI_layers

def knn(inp):
    pass

