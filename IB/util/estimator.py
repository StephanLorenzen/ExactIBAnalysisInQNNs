import numpy as np

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
    def __call__(self, A, Y):
        raise Exception("Not implemented!")

class BinningEstimator(BaseEstimator):
    def __init__(self, strategy="uniform", n_bins=30, bounds="neuron"):
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

    def __call__(self, A, Y):
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
       
            MI_XT = it.entropy(T)
            MI_TY = it.mutual_information(T,Y)
            MI_layers.append((MI_XT,MI_TY))
        
        return MI_layers

class QuantizedEstimator(BaseEstimator):
    def __init__(self, bounds="neuron"):
        assert(bounds in ("layer","neuron"))
        super().__init__("Quantized computation, "+bounds,"quantized_"+bounds)
        self.bounds = bounds
    
    def __call__(self, A, Y):
        MI_layers = []
        for layer in A:
            # Compute binning neuron wise
            if self.bounds=="neuron":
                # Compute binning neuron wise
                T = np.apply_along_axis(binning.quantized, 0, layer)
            elif self.bounds=="layer":
                # Compute binning layer wise
                T = binning.quantized(layer)
            MI_XT = it.entropy(T)
            MI_TY = it.mutual_information(T,Y)
            MI_layers.append((MI_XT,MI_TY)) 
        return MI_layers
    

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
    bs = params.get("binning_strategy", "neuron") # neuron, layer, fixed 
    n_bins = params.get("n_bins", 30)
    MI_layers = []
    for layer in A:
        if bs=="neuron":
            # Compute binning neuron wise
            T = np.apply_along_axis(lambda nr: bin_func(nr, n_bins), 0, layer)
        elif bs=="layer":
            # Compute binning layer wise
            T = bin_func(layer,n_bins)
        else: # bs==fixed
            assert("bins" in params)
            # Use precomputed bins
            T = np.digitize(layer, params["bins"])
        
        MI_XT = discrete.entropy(T)
        MI_TY = discrete.mutual_information(T,Y)
        MI_layers.append((MI_XT,MI_TY))
    return MI_layers

