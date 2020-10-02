import functools
import numpy as np
from numba import jit
from scipy import constants
from scipy.signal import find_peaks
from multiprocessing import Pool
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
#TODO: Implement fitting with lmfit
import lmfit as lm

from . import data as _d

# Pretty Plotting
import os
plt.style.use(os.path.join(os.path.dirname(__file__),"style.mplstyle"))

pi = constants.pi
c  = constants.c

##################
# Analytic Error #
##################
@jit(nopython=True)
def errf(dL, r, fm, m, lamb, a, phi, theta):
    dl = (4 * pi / lamb) * dL
    lm = 2 * pi * fm * m * lamb / c
    poly1=(24 - 12*dl**2 + dl**4)
    t1 = ((lm**2 *
          (-24 + lm**2)**2 * r**2 * (-1 + r**2) * (3 * dl * r**2 *
            (-1 + r**2) * (6 *
                (-3 + lm**2 + 2 * r**2 + r**4) -
                dl**2 * (-3 + lm**2 + 8 * r**2 + r**4)) +
            3 * a * dl * (-6 + dl**2 + 4 *
                (-3 + 2 * dl**2) * r**2 + 2 * (-6 + dl**2) * (-3 + lm**2) * r**4 +
                4 * (-3 + 2 * dl**2) * r**6 + (-6 + dl**2) * r**8) * np.cos(phi) -
            a * (-1 + r**4) * (9 *
                (-2 + dl**2) + (36 + dl**2 * (-6 + dl**2)**2 - 18 * lm**2) * r**2 +
                9 * (-2 + dl**2) * r**4)  * np.sin(phi))) /
          (32. * (1 + (-2 + dl**2) * r**2 + r**4) * (18 + r**2 * (6 * dl * lm * (-6 + lm**2) - dl**3 * lm * (-6 + lm**2) - 9 * dl**2 * (-2 + lm**2) + 18 * (-2 + lm**2 + r**2))) *
          (18 + r**2 * (-6 * dl * lm * (-6 + lm**2) + dl**3 * lm * (-6 + lm**2) - 9 * dl**2 * (-2 + lm**2) + 18 * (-2 + lm**2 + r**2)))))

    t2 = ((-7776 * (lm - lm**3 / 6.) * r**2 * (-1 + r**2) * ((2 * dl - (4 * dl**3) / 3.) * r**2 * (r**4 + a * np.cos(phi)) +
         (dl - dl**3 / 6.) * ((-2 + lm**2) * r**2 * (-1 + r**2) - (poly1 * (r**4 + a * r**6 * np.cos(phi))) / 12. + (-1 + r**4) * (r**2 - r**4 + a * (1 + r**4) * np.cos(phi))) -
         a * (dl - dl**3 / 6.)**2 * r**2 * (1 + r**4) * np.sin(phi) + a * (-((-24 + 12 * dl**2 - dl**4 + 24 * r**2) * (-24 + poly1 * r**2) * (1 + r**4)) / 576. +
         2 * (1 - lm**2 / 2.) * (r**2 + (-2 + dl**2 - dl**4 / 12.) * r**4 + r**6)) * np.sin(phi))) /
         ((12 - poly1 * r**2 + 12 * r**4) * (18 + r**2 * (6 * dl * lm * (-6 + lm**2) - dl**3 * lm * (-6 + lm**2) - 9 * dl**2 * (-2 + lm**2) + 18 * (-2 + lm**2 + r**2))) *
         (18 + r**2 * (-6 * dl * lm * (-6 + lm**2) + dl**3 * lm * (-6 + lm**2) - 9 * dl**2 * (-2 + lm**2) + 18 * (-2 + lm**2 + r**2)))))

    err = t1*np.cos(theta) + t2*np.sin(theta)
    return err

#########################
# Analytic Transmission #
#########################
@jit(nopython=True)
def transf(dL, r, fm, m, lamb, pc):
    dl = (4 * pi / lamb) * dL
    lm = 2 * pi * fm * m * lamb / c
    poly1= (24 - 12*dl**2 + dl**4)
    # Carrier Trans
    cr = (12*pc*(-1 + r**2)**2)/(12 - poly1*r**2 + 12*r**4)
    # Sideband Trans
    sb = (-((-1 + pc)*(-1 + r**2)**2 * (288 - poly1*(24 - 12*lm**2 + lm**4)*r**2 + 288*r**4))/
           (72.*(2 - (4 - 2*dl**2 + 4*dl*lm + (-2 + dl**2)*lm**2)*r**2 + 2*r**4) *
                (2 + r**2*(4*dl*lm - dl**2*(-2 + lm**2) + 2*(-2 + lm**2 + r**2)))))
    tr = cr+sb
    return tr

#############
# Normalize #
#############
@jit(nopython=True)
def norm(vals):
    return vals/np.max(np.abs(vals))

###########################
# Parametric Minimization #
###########################
@jit(nopython=True, parallel=True) # We want this to be as fast as possible, so let's JIT it with parallelization
def min_dist(pair,es,ts):
    e = pair[0]
    t = pair[1]
    return np.min(np.hypot((e-es),(t-ts)))

def min_dists(pairs, es,ts):
    func = functools.partial(min_dist, es=es, ts=ts)
    with Pool(10) as p:
        return np.array(p.map(func, pairs))

#########################
# Simple Fitting Funcs. #
#########################
def fano(x, amp, slope, width, center):
    hw = width/2
    deltax = x-center
    return (amp + slope * deltax) * hw**2 /((hw)**2 + deltax**2)

def triple_fan(x, splitting, amp, slope, center, linewidth, ps, offset):
    shift = splitting/2
    carrier = fano(x, amp, slope, linewidth, center)
    sidebands = ps * (fano(x, amp, slope, linewidth, center+shift) 
                      + fano(x, amp, slope, linewidth, center-shift))
    return offset + carrier + sidebands

def lorenz(x, amp, width, center):
    hw = width/2
    deltax = x-center
    return (amp) * hw**2 /((hw)**2 + deltax**2)

def triple_lor(x, splitting, amp, center, linewidth, ps, offset):
    shift = splitting/2
    carrier = lorenz(x, amp, linewidth, center)
    sidebands = ps * (lorenz(x, amp, linewidth, center+shift) 
                      + lorenz(x, amp, linewidth, center-shift))
    return offset + carrier + sidebands

####################
# Sideband Fitting #
####################
def fit_triple(filename,ax,func,mod_freq):
    print("Fitting sideband data in %s" % filename)
    data = _d.read_csv(filename, delimiter='\t',skip_header=27)
    xs = data[:,0]
    ys = data[:,1]

    # Getting main peak
    peak=find_peaks(ys, height=(np.mean(ys)+0.1*np.std(ys)),distance=50000)[0][0]

    # Rough guessing side peaks
    split_left = np.argmin(np.abs(ys[:peak] - np.max(ys[:peak])/3))
    split_right = np.argmin(np.abs(ys[peak:] - np.max(ys[peak:])/3))

    # Guesses
    offset = np.mean(ys)/2
    center = xs[peak]
    split = (xs[split_right]-xs[split_left])
    amp = max(ys) - offset
    guesses = [split*0.5,amp/1.5,center,0.0001,0.1,offset]
    sigma = min(np.diff(data[:,1]))

    # Fitting
    p_opt, p_cov = curve_fit(func,
                             data[:,0],data[:,1],
                             p0=guesses,
                             sigma=sigma*np.ones(data[:,1].size),
                             maxfev=2000)
    
    # Computing results
    errors = np.sqrt(np.diag(p_cov))
    chisqr = np.sum(((ys - triple_lor(xs,*p_opt))/sigma)**2)/(len(ys)-len(guesses))
    ax.plot(xs, triple_lor(xs,*p_opt))
    if chisqr > 2.0:
        print("Chi-Square from triplet fit is greater than 2!")
    lw = p_opt[3] / p_opt[0] * mod_freq # MHz
    print("Linwidth = %.2f MHz (Chisq = %.2f)" % (np.abs(lw),chisqr))
    return lw

#####################
# WhiteLight Length #
#####################
def white_length(filename, plot=False, disp=False, 
                 wlmin=600, wlmax=650, dist=50, threshold=None, ratio=0.05, **kwargs):
    wl_data = _d.read_csv(filename)
    wavelength = wl_data[:,0]
    counts = np.max(wl_data[:,1:17],axis=1)
    if threshold is None:
        threshold = np.mean(counts) + (np.max(counts) * ratio)

    bounds = np.logical_and(wavelength >= wlmin, wavelength <= wlmax)
    wavelength = wavelength[bounds]
    counts = counts[bounds]

    peaks = find_peaks(counts, threshold=threshold, distance=dist, **kwargs)
    peak_wl = wavelength[peaks[0]]
    peak_freq = c/(peak_wl * 1E-9)
    fsrs = np.diff(peak_freq[::-1])
    lengths = c/(2*fsrs)
    #TODO: USE THAT NCR PACKAGE FOR UNCERTAINTIES
    length = np.mean(lengths) * 1E6 # um
    fsr = np.mean(fsrs) / 1E6 # MHz
    if plot == True:
        plt.figure()
        plt.plot(wavelength,counts)
        plt.vlines(peak_wl,np.min(counts),np.max(counts)+10)
        plt.show(block=False)
    elif plot != False:
        plot.plot(wavelength, counts)
        plot.vlines(peak_wl,np.min(counts),np.max(counts)+10)
    if disp:
        print(filename)
        print("\tCavity length is: %s um" % length)
        print("\tFSR is: %s MHz" % fsr)
    return length, fsr
