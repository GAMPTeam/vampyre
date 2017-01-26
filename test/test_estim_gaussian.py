"""
    test_estim_gaussian.py

    Run battery of tests associated for the module `vampyre.estim.gaussian`.
"""
import numpy as np
from numpy.random import uniform, normal
from vampyre.common.utils import TestException
from vampyre.common.utils import repeat_axes, repeat_sum
from vampyre.estim.base import Estim
from vampyre.estim.gaussian import GaussEst

def test_gauss_est(zshape=(1000,10), verbose=False, tol=0.1):
    """

    
    test_gauss_est [vampyre.estim.gaussian.GaussEst]

    Unit test for the Gaussian estimator class :class:`GaussEst`.
    
    The test works by creating synthetic Gaussian variables :math:`z`, and 
    measurements 
    
    :math:`r = z + w,  w \sim {\mathcal N}(0,\\tau_r)`
    
    Then, the :func:`Gaussian.est` and :func:`Gaussian.est_init` methods 
    are called to see if :math:`z` with the expected variance.
                
    :param zshape: Shape of :param:`z`.  The parameter can be an 
       arbitrary :code:`ndarray'.
    :param Boolean verbose:  Flag indicating if results are to be printed
    :param tol:  Tolerance for raising an error
    :param Boolean raise_exception: raise an :class:`TestException` if 
       test fails.
    """    

    # Generate synthetic data with random parameters
    zvar = uniform(0, 1, 1)[0]
    rvar = uniform(0, 1, 1)[0]
    zmean = normal(0, 1, 1)[0]
    z = zmean + normal(0, np.sqrt(zvar), zshape)
    r = z + normal(0, np.sqrt(rvar), zshape)
    
    # Construct estimator
    est = GaussEst(zmean, zvar, zshape, var_axes='all')

    # Inital estimate
    zmean1, zvar1 = est.est_init()
    zerr = np.mean(np.abs(z-zmean1)**2)
    # print("[test_gauss_est] Initial -- True: {0:f}, Est:{1:f}".format(zerr, zvar1))
    
    if np.abs(zerr-zvar1) > tol*np.abs(zerr):
        raise TestException("Initial estimate Gaussian error "+ 
                            "does not match predicted value.")
    
    # Posterior estimate
    zhat, zhatvar, cost = est.est(r, rvar, return_cost=True)
    zerr = np.mean(np.abs(z-zhat)**2)
    # if verbose:
        # print("[test_gauss_est] Posterior -- True: {0:f}, Est:{1:f}".format(zerr,zhatvar))
    if np.abs(zerr-zhatvar) > tol*np.abs(zhatvar):
        raise TestException("Posterior estimate Gaussian error "+ 
                            "does not match predicted value.")    
