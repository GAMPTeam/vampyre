"""
test_lin_two.py:  Test suite for the linear estimator :class:`LinEstimTwo`
"""
from __future__ import print_function

import unittest
import numpy as np

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp

                      
def binary_test(zshape=(1000,10), verbose=False, rvar=None, tol=0.1):
    """
    Unit test for the :class:`HardThreshEst` class
    
    The test works by creating synthetic distribution, creating an i.i.d.
    matrix :math:`z` with Gaussian components and then taking quantized
    values :math:`y = (z > t)` where `t` is a random threshold    
    Then, the estimation methods are called to see if 
    the measured error variance matches the expected value.
            
    :param zshape: shape of :math:`z`
    :param Boolean verbose:  prints results.  
    :parma rvar:  Gaussian variance.  :code:`None` will generate a random
       value
    :param tol:  Tolerance for test test to pass
    """    
    
    # Generate random parameters
    if rvar == None:
        rvar = 10**(np.random.uniform(-1,1,1))[0]
    
    # Generate data
    r = np.random.normal(0,1,zshape)
    z = r + np.random.normal(0,np.sqrt(rvar),zshape)
    thresh = 0 #np.random.uniform(-1,1,1)[0]   
    y = (z > thresh)
    
    # Create estimator
    est = vp.estim.BinaryQuantEst(y,shape=zshape,thresh=thresh,var_axes='all')
    
    # Run the initial estimate.  In this case, we just check that the 
    # dimensions match
    zmean, zvar, cost = est.est_init(return_cost=True)
    if not (zmean.shape == zshape):
        raise vp.common.TestException("Initial shape are not correct")
    if not np.isscalar(zvar):
        if zvar.shape != (1,):
            raise vp.common.TestException("Initial shape of variance is incorrect")
            
                    
    # Get posterior estimate
    zhat, zhatvar, cost = est.est(r,rvar,return_cost=True)
    
    # Measure error
    zerr = np.mean(np.abs(zhat-z)**2)
    if verbose:
        print("err: true: {0:12.4e} est: {1:12.4e}".format(zerr,zhatvar) )
    if (np.abs(zerr-zhatvar) > tol*np.abs(zerr)):
        raise vp.common.TestException("Posterior estimate for error "+ 
           "does not match predicted value")
        

class TestCases(unittest.TestCase):
    def test_binary_quant(self):
        verbose = False
        binary_test(verbose=verbose)
        
        
if __name__ == '__main__':    
    unittest.main()
    
