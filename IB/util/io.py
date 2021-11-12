import os
import math
import numpy as np
import pandas as pd

def load_MI_repeats(path, est="binning_uniform_30"):
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
    
    return np.array(repeats)

def load_MI(path, est=None):
    repeats = load_MI_repeats(path,est)
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

    train_ci = 1.96*train_std/math.sqrt(len(train_accs))
    test_ci  = 1.96*test_std/math.sqrt(len(test_accs))
    return (train_mean, train_std, train_ci), (test_mean, test_std, test_ci)


def load_activations(path):
    if not os.path.isdir(path):
        raise Exception("Directory does not exists '"+path+"'")
    def _zp(val):
        val = str(val)
        return "0"*(3-len(val)) + val
    l = None
    files = []
    act_path = path+"activations/"
    for f in os.listdir(act_path):
        files.append(f)

    max_acts, min_acts = [],[]
    files.sort()
    for f in files:
        cnt = int(f[:3])-1
        if "_max" in f:
            max_acts.append(pd.read_csv(act_path+f,index_col="epoch").values)
        else:
            min_acts.append(pd.read_csv(act_path+f,index_col="epoch").values)
    

    max_acts, min_acts = np.array(max_acts), np.array(min_acts)
    max_mean, max_std  = np.mean(max_acts,axis=0), np.std(max_acts,axis=0)
    min_mean, min_std  = np.mean(min_acts,axis=0), np.std(min_acts,axis=0)

    return (max_mean, max_std), (min_mean, min_std)

def load_activations_2D(path):
    assert os.path.isdir(path), "Directory does not exists '"+path+"'"
    def _zp(val):
        val = str(val)
        return "0"*(3-len(val)) + val
    l = None
    dfs = []
    act_path = path+"activations/2D/"
    for f in os.listdir(act_path):
        es = f.split(".")[0]
        dfs.append((es,pd.read_csv(act_path+f, index_col="i")))
    return dfs


