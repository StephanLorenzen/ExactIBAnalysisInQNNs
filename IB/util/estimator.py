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
    def __call__(self, A, Y, X=None):
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
            MI_XT = it.entropy(T)
            MI_TY = it.mutual_information(T,Y)
            MI_layers.append((MI_XT,MI_TY))
        
        return MI_layers

class QuantizedEstimator(BaseEstimator):
    def __init__(self, bounds="neuron", bits=8):
        assert(type(bounds)==tuple or bounds in ("layer","neuron"))
        bstr = bounds if type(bounds)==str else "fixed"
        bstr += "_"+str(bits)+"_bits"
        super().__init__("Quantized computation, "+bstr,"quantized_"+bstr)
        self.bounds = bounds
        self.bits = bits
    
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
            MI_XT = it.entropy(T)
            MI_TY = it.mutual_information(T,Y)
            MI_layers.append((MI_XT,MI_TY)) 
        prev = 100
        #for _, miy in MI_layers:
        #    if miy > prev+10**-5:
        #        print("Detected non-monotone MI(Y,T)")
                #print(MI_layers)
                #assert(False)
                #import pdb; pdb.set_trace()
        #    prev = miy
        return MI_layers
    

def _knn_H(k, X, norm_p=math.inf):
    assert(norm_p==math.inf) # Not implemented for others
    N,d = X.shape
    nn = NN(n_neighbors=k+1, p=norm_p).fit(X)
    dist, _ = nn.kneighbors(X)
    eps = dist[:,-1] # distance to k+1 (=k without self) neighbor
    return -digamma(k)+digamma(N)+(d/N)*np.sum(np.log(2*eps))

def _knn_I(k, X, Y, norm_p=math.inf):
    assert(norm_p==math.inf) # Not implemented for others
    assert(len(X)==len(Y))
    Y = Y.reshape((-1,1))
    N = len(X)

    Z = np.concatenate((X,Y), axis=1)
    
    # Find distance in Z-space
    nn = NN(n_neighbors=k+1, p=norm_p).fit(Z)
    dist, _ = nn.kneighbors(Z)
    eps = dist[:,-1] # distance to k+1 (=k without self) neighbor
    nn,dist=None,None

    # For each neighbor, project X/Y onto x-plane/y-plane.
    nnX, nnY = NN(p=norm_p).fit(X), NN(p=norm_p).fit(Y)
    def _find_nx(i):
        x,ex = X[i],eps[i]
        ind = nnX.radius_neighbors([x],radius=ex,return_distance=False)
        return ind[0].shape[0]
    def _find_ny(i):
        y,ey = Y[i],eps[i]
        ind = nnY.radius_neighbors([y],radius=ey,return_distance=False)
        return ind[0].shape[0]

    # Find n_x and n_y
    nx = np.array(list(map(_find_nx, range(N))))
    ny = np.array(list(map(_find_ny, range(N))))
    # Add 1
    nx = nx+1
    ny = ny+1
    # Apply digamma
    nx = np.array(list(map(digamma, nx)))
    ny = np.array(list(map(digamma, ny)))

    res = digamma(k)-(nx+ny).mean()+digamma(N)
    #import pdb; pdb.set_trace() 
    return res

class KNNEstimator(BaseEstimator):
    def __init__(self, k=10):
        super().__init__("kNN computation, k="+str(k),"knn_"+str(k))
        self.k = k

    def __call__(self, A, Y, X=None):
        MI_layers = []
        for layer in A:
            MI_XT = _knn_H(self.k, layer)
            MI_TY = _knn_I(self.k, layer, Y)
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

