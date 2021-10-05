import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors as NN
import math

from . import binning, it

class BaseEstimator:
    def __init__(self, name, fname):
        self._name  = name
        self._fname = fname
    def __str__(self):
        return self._name
    def dir(self):
        return self._fname
    def setup(self, activations):
        return # Implement if needed
    def require_setup(self):
        return True; # Set to false if not needed
    def __call__(self, A, Y, X=None):
        raise Exception("Not implemented!")

class BinningEstimator(BaseEstimator):
    def __init__(self, strategy="uniform", n_bins=30, bounds="neuron", use_mean=False):
        assert(strategy in ("uniform","adaptive"))
        assert(type(n_bins)==int)
        assert(type(bounds)==tuple or bounds in ("global","layer","neuron"))
        assert(not (type(bounds)==tuple and strategy=="adaptive"))

        bbstr = "fixed" if type(bounds)==tuple else bounds
        super().__init__(
            "Binning estimator, "+strategy+", "+str(n_bins)+" bins, "+bbstr,
            "binning_"+strategy+"_"+str(n_bins)+"_"+bbstr 
        )
        self.strategy = strategy
        self.n_bins   = n_bins
        self.fixed    = type(bounds)==tuple
        self.bounds   = bounds
        
        self.bins     = None
        self.mean     = use_mean

    def setup(self, activations):
        if self.fixed:
            assert(self.strategy=="uniform") # Double check
            lw,up = self.bounds
            self.bins = binning.uniform_bins(self.n_bins, lower=lw, upper=up)
        elif self.bounds=="global":
            als = []
            for epoch in activations:
                als.append(np.concatenate(list(map(lambda x: x.flatten(), epoch))))
            activations = np.concatenate(als)
            if self.strategy == "uniform":
                self.bins = binning.uniform_bins(self.n_bins, values=activations)
            else:
                self.bins = binning.adaptive_bins(self.n_bins, activations)
        # else:
        #   no setup needed

    def __call__(self, A, Y, X=None):
        MI_layers = []
        for layer in A:
            bin_func = {"uniform":binning.uniform,"adaptive":binning.adaptive}[self.strategy]
            if self.bounds=="neuron":
                # Compute binning neuron wise
                T = np.apply_along_axis(lambda nr: bin_func(self.n_bins, nr), 0, layer)
            elif self.bounds=="layer":
                # Compute binning layer wise
                T = bin_func(self.n_bins, layer)
            else: # fixed or self.bounds==global
                assert(self.bins is not None)
                # Use precomputed bins
                T = np.digitize(layer, self.bins)
            MI_XT = it.split_entropy(T) if self.mean else it.entropy(T)
            MI_TY = it.split_mutual_information(T,Y) if self.mean else it.mutual_information(T,Y)
            MI_layers.append((MI_XT,MI_TY))
        
        return MI_layers

class QuantizedEstimator(BaseEstimator):
    def __init__(self, bounds="neuron", bits=8, use_mean=False):
        assert(type(bounds)==tuple or bounds in ("layer","neuron"))
        bstr = bounds if type(bounds)==str else "fixed"
        bstr += "_"+str(bits)+"_bits"
        super().__init__("Quantized computation, "+bstr,"quantized_"+bstr)
        self.bounds = bounds
        self.bits = bits
        self.mean = use_mean
   
    def require_setup(self):
        return False;

    def __call__(self, A, Y, X=None):
        MI_layers = []
        for i,layer in enumerate(A):
            # Compute binning neuron wise
            if type(self.bounds)==tuple:
                # Compute binning with fixed edges
                b = self.bounds if i<len(A)-1 else (0,1)
                T = binning.quantized(layer, lower=b[0], upper=b[1], bits=self.bits)
            elif self.bounds=="neuron":
                # Compute binning neuron wise
                T = np.apply_along_axis(lambda l: binning.quantized(l, bits=self.bits), 0, layer)
            elif self.bounds=="layer":
                # Compute binning layer wise
                T = binning.quantized(layer, bits=self.bits)
            MI_XT = it.split_entropy(T) if self.mean else it.entropy(T)
            MI_TY = it.split_mutual_information(T,Y) if self.mean else it.mutual_information(T,Y)
            MI_layers.append((MI_XT,MI_TY)) 
        return MI_layers
    
