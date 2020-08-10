import functools
import numpy as np
from numba import jit
from scipy import constants
from multiprocessing import Pool

pi = constants.pi
c  = constants.c

##################
# Error Function #
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
# Transmission Function #
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