"""
    test_common_utils.py

    Run battery of tests associated for the module `vampyre.common.utils`.
"""
import numpy as np
from vampyre.common.utils import TestException
from vampyre.common.utils import repeat_axes, repeat_sum

def test_repeat_axes(tol=1e-8):
    """
    test_repeat_axes() [vampyre.common.utils.repeat_axes]

    :param tol:  Tolerance for passing test.
    """    
    ushape = (2, 2, 3)
    rep_axes = (1, 2)
    u = np.random.uniform(0, 1, size=ushape)
    uavg = np.mean(u, axis=rep_axes)
    urep = repeat_axes(uavg, shape=ushape, rep_axes=rep_axes)
    
    # Makse sure that the sums are the same
    difference = 0
    for i in range(ushape[0]):
        sum0 = np.sum(u[i, :, :])
        sum1 = np.sum(urep[i, :, :])
        difference += np.abs(sum1-sum0)
    
    # print("\n[test_repeat_axes] Difference in sums: {0:f}".format(difference))
                
    if difference > tol:
        raise TestException("[test_repeat_axes] Sum along repeated matrix is not conistent.")
