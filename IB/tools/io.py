import os
import numpy as np
import pandas as pd

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


def load_accuracy(path):
    if not os.path.isdir(path):
        raise Exception("Directory does not exists '"+path+"'")
    def _zp(val):
        val = str(val)
        return "0"*(3-len(val)) + val
    l = None
    files = []
    acc_path = path+"accuracy/"
    for f in os.listdir(acc_path):
        files.append(f)

    train_accs, test_accs = [],[]
    files.sort()
    for f in files:
        cnt = int(f[:3])-1
        assert(cnt==len(train_accs))
        repeat = pd.read_csv(acc_path+f,index_col="epoch")
        train_accs.append(repeat["train_acc"].values)
        test_accs.append(repeat["test_acc"].values)
    
    train_accs, test_accs = np.array(train_accs), np.array(test_accs)
    train_mean, train_std = np.mean(train_accs,axis=0), np.std(train_accs,axis=0)
    test_mean,  test_std  = np.mean(test_accs,axis=0),  np.std(test_accs,axis=0)

    return (train_mean, train_std), (test_mean, test_std)
