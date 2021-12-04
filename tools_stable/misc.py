from numba import njit
import math
#place to implement miscellaneous methods


#numba does not support scipy.stats.norm.cdf soo i've implemented this
@njit
def normcdf(x):
    return math.erfc(-x / math.sqrt(2.0)) / 2.0