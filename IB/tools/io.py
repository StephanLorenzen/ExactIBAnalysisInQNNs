import os
import numpy as np

def load_MI(path, est="binning_uniform_30"):
    if not os.path.isdir(path):
        raise Exception("Directory does not exists '"+path+"'")
    repeats = []
    def _zp(val):
        val = str(val)
        return "0"*(3-len(val)) + val
    l = None
    files = []
    mi_path = path+"mi/"+est+"/"
    for f in os.listdir(mi_path):
        files.append(f)

    files.sort()
    for f in files:
        cnt = int(f[:3])-1
        assert(cnt==len(repeats))
        epoch = np.genfromtxt(mi_path+f)
        l = len(epoch) if l is None else l
        assert(len(epoch)==l)
        repeats.append(epoch)
    
    repeats = np.array(repeats)
    nmean   = np.mean(repeats,axis=0)
    nstd    = np.std(repeats,axis=0)
    
    num_epochs, num_layers = nmean.shape
    num_layers //= 2

    mean, std = [], []
    for ms,ss in zip(nmean,nstd):
        mean.append([(ms[2*i],ms[2*i+1]) for i in range(num_layers)])
        std.append([(ss[2*i],ss[2*i+1]) for i in range(num_layers)])

    return mean, std
