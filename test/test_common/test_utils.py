import sys, os
sys.path.insert(0, '.')

import numpy as np
from vampyre.common.utils import repeat_axes

def test_repeat(verbose=False, tol=1e-8, raise_exception=True):
    """
    Unit test for the repeat_mult function.

    :param Boolean verbose:  Print test results    
    :param tol:  Tolerance for passing test
    :param Boolean raise_exception:  Raises an error on test failure.  This 
        can be caught in the unit test dispatcher.
    """    
    ushape = (2,2,3)
    rep_axes = (1,2)
    u = np.random.uniform(0,1,size=ushape)
    uavg = np.mean(u, axis=rep_axes)
    urep = repeat_axes(uavg,shape=ushape,rep_axes=rep_axes)
    
    # Makse sure that the sums are the same
    d = 0
    for i in range(ushape[0]):
        sum0 = np.sum(u[i,:,:])
        sum1 = np.sum(urep[i,:,:])
        d += np.abs(sum1-sum0)
    if verbose:
        print("total diff= {0:f}".format(d))
                
    if (d > tol) and raise_exception:
        raise TestException("Sum along repeated matrix is not conistent")