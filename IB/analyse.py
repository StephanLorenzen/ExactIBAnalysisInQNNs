import numpy as np

from .tools import plot


def plot_IB_plane(path):
    iters = []
    def _zp(val):
        val = str(val)
        return "0"*(3-len(val)) + val
    for i in range(1,5+1):
        iters.append(np.genfromtxt(path+_zp(i)+"_mi.txt"))
    iters = np.array(iters)
    iters = np.mean(iters,axis=0)

    plot.information_plane(iters)
