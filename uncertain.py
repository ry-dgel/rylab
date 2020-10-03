import numpy as np
import metrolopy as mp

def from_list(xs, weights=None, unit=None):
    """
    Given a list of x values, returns a gummy with mean center
    and std error uncertainty. The calculation of the uncertainty is 
    based on Sect. 4.2 of the BIPM's 
    "Guide to the Expression of Uncertainty in Measurement"

    Parameters
    ----------
    xs : [number]
        The values to convert into a single mean with uncertainty.
    weights : [number], optional
        The weights to be considered for each x value when averaging
        and taking the standard deviation., by default None
    unit : str, optional
        The unit of the value and uncertainty to pass to gummy, by default None

    Returns
    -------
    gummy
        The gummy object containing the mean and uncertainty.
    """
    if weights is not None:
        if len(xs) != len(weights):
            raise RuntimeError("xs and weights must be of same length")

    mean = np.average(xs, weights=weights)
    # Calculate standard deviation, weighted if needed.
    # np.average divides sum of squared deviations by n. To properly compute
    # variance this should be (n-1)
    var = np.average((xs-mean)**2, weights=weights)*len(xs)/(len(xs)-1)
    # Standard error on mean is sqrt of (variance divided by n)
    stder = np.sqrt(var/len(xs))

    return mp.gummy(mean, stder, unit=unit)