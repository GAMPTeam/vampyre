"""
test_gaussian.py:  Test suite for the Gaussian estimator class :class:`GaussEst`
"""
from __future__ import print_function

import unittest
import numpy as np

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp

def gauss_test(zshape=(1000,10), verbose=False, tol=0.1):
    """
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
    zvar =  np.random.uniform(0,1,1)[0]
    rvar = np.random.uniform(0,1,1)[0]
    zmean = np.random.normal(0,1,1)[0]
    z = zmean + np.random.normal(0,np.sqrt(zvar),zshape)
    r = z + np.random.normal(0,np.sqrt(rvar),zshape)
    
    # Construct estimator
    est = vp.estim.GaussEst(zmean,zvar,zshape,var_axes='all')

    # Inital estimate
    zmean1, zvar1 = est.est_init()
    zerr = np.mean(np.abs(z-zmean1)**2)
    fail = (np.abs(zerr-zvar1) > tol*np.abs(zerr))
    if verbose or fail:
        print("Initial:      True: {0:f} Est:{1:f}".format(zerr,zvar1))
    if (fail):
        raise vp.common.TestException("Initial estimate Gaussian error "+ 
           "does not match predicted value")
    
    # Posterior estimate
    zhat, zhatvar, cost = est.est(r,rvar,return_cost=True)
    zerr = np.mean(np.abs(z-zhat)**2)
    fail = (np.abs(zerr-zhatvar) > tol*np.abs(zhatvar))
    if verbose or fail:
        print("Posterior:    True: {0:f} Est:{1:f}".format(zerr,zhatvar))
    if fail:
        raise vp.common.TestException("Posterior estimate Gaussian error "+ 
           "does not match predicted value")   

class TestCases(unittest.TestCase):
    def test_gauss(self):
        gauss_test()
        
if __name__ == '__main__':    
    unittest.main()
    
