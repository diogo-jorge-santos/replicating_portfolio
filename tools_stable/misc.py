from numba import njit
import math
import numpy as np
import scipy.stats as sp
# place to implement miscellaneous methods


# numba does not support scipy.stats.norm.cdf soo i've implemented this
@njit
def normcdf(x):
    return math.erfc(-x / math.sqrt(2.0)) / 2.0

#functions to be used in the risk return function
def VaR(result, alpha=0.05):
    return np.quantile(result,alpha)

def ES(result, alpha=0.05):
    return np.mean(result[result<=np.quantile(result,alpha)])


def mean_CI(result, alpha=0.001):
    m, se = np.mean(result), sp.sem(result)
    h = se * sp.norm.ppf(1-alpha/2)
    return m, m-h, m+h

def sd_CI(result, alpha=0.001):
    var=np.var(result,ddof=1)
    lower_ci= (result.shape[0]-1)*var/sp.chi2.ppf(1-(1-alpha/2),df=result.shape[0]-1)
    higher_ci = (result.shape[0]-1)*var/sp.chi2.ppf(1-alpha/2,df=result.shape[0]-1)
    return var**0.5,lower_ci**0.5,higher_ci**0.5
