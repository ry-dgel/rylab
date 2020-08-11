import numpy as np
import spinmob as sp
import matplotlib.pyplot as plt

import os
plt.style.use(os.path.join(os.path.dirname(__file__),"style.mplstyle"))

########
# PSDs #
########
# Generate a PSD from a time trace
def take_psd(filename, skip=21):
    data = np.genfromtxt(filename, delimiter=',', skip_header=21)
    f, psd = sp.fun.psd(data[:,0], data[:,1], window='hanning', rescale=True)
    return f,psd

def raw_psd(data):
    return sp.fun.psd(data[:,0], data[:,1], window='hanning', rescale=True)

# Load a PSD file (".A!") from sillyscope
# skip is the length of the header in the file
# idx_offset is added to the y-data column index
# for choosing from multiple channels
def load_psd(filename, skip=53, idx_offset=0, VtoL=None):
    data = np.genfromtxt(filename, skip_header=skip)
    f = data[:,0]
    y = data[:,1+idx_offset]
    if VtoL is not None:
        y = y* VtoL**2
    return f,y

# Coarsens a psd exponentially with spinmob,
# level is passed to function
def coarse_psd(x,y,level=1.01):
    return sp._functions.coarsen_data(x,y,level=level,exponential=True)

# Given a PSD file and a plot axis, plots the raw data
# and the coarsened data on top. Returns raw data from file as well.
# linear keyword allows axes to be set to linear, otherwise defaults to loglog
def plot_psd(filename, ax, label=None, idx_offset=0,level=1.04, linear=None, VtoL=None):
    if label is None:
        label = filename
    f,y = load_psd(filename, 53, idx_offset, VtoL)
    l = ax.loglog(*coarse_psd(f,y,level=level), label=label, zorder=10)
    ax.loglog(f,y,color=l[0].get_color(),linestyle=':',alpha=0.5,zorder=5)
    if linear is not None:
        if "x" in linear.lower():
            ax.set_xscale("linear")
        if "y" in linear.lower():
            ax.set_yscale("linear")

    return f,y

def raw_plot_psd(f, y, ax, label=None, level=1.04, linear=None, alpha=0.5, smooth=True, raw=True):
    if smooth:
        l = ax.loglog(*coarse_psd(f,y,level=level), label=label, zorder=10)
        if raw:
            ax.loglog(f,y,color=l[0].get_color(),linestyle=':',alpha=alpha,zorder=5)
    elif raw:
        ax.loglog(f,y,label=label,zorder=10)
    if linear is not None:
        if "x" in linear.lower():
            ax.set_xscale("linear")
        if "y" in linear.lower():
            ax.set_yscale("linear")
    return f,y

# Adds a fancy legend above the plot to avoid covering data.
def fancy_leg(fig):
    fig.legend(prop={'size': 8},loc='upper center', 
               bbox_to_anchor=(0.5, 0.9), ncol=4, fancybox=True)