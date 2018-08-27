"""
test_relu.py:  Test suite for the ReLU estimator class :class:ReLUEstim`
"""
from __future__ import print_function
from __future__ import division

import unittest
import numpy as np

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp



def logistic_test(zshape=(100,10), rvar=1, tol=1, verbose=False):
    """
    Unit test for the logistic estimator.  
    Generates random data with a logistic model and then estimates the input
    logit :code:`z`.    
    
    :param zshape:  shape of the data :code:`z`
    :param rvar: prior variance on :code:`r`
    :param tol:  tolerance on estimation error.  This should be large since we
        are using MAP instead of MMSE estimation so the error variance
        is not exact
    :param verbose:  print results
    """

    # Create random data    
    z = np.random.normal(0,1,zshape)
    r = z + np.random.normal(0,np.sqrt(rvar),zshape)
    pz = 1/(1+np.exp(-z))
    u = np.random.uniform(0,1,zshape)
    y = (u < pz)
    
    # Create an estimator
    est = vp.estim.LogisticEst(y=y,var_axes='all',max_it=100)
    
    # Run the estimator 
    zhat, zhatvar = est.est(r,rvar)
    
    # Compare the error
    zerr = np.mean((z-zhat)**2)
    rel_err = np.maximum( zerr/zhatvar, zhatvar/zerr)-1
    fail = (rel_err > tol)
    
    if fail or verbose:
        print("Error:  Actual: {0:12.4e} Est: {1:12.4e} Rel: {2:12.4e}".format(\
            zerr, zhatvar, rel_err))
    if fail:
        raise vp.common.TestException("Estimation error variance"+\
            " does not match predicted value")   

class TestCases(unittest.TestCase):
    def test_logistic(self):
        verbose = False        
        logistic_test(rvar=0.1, verbose=verbose,tol=0.1)        
        logistic_test(rvar=10, verbose=verbose,tol=0.5)        
        
if __name__ == '__main__':    
    unittest.main()    
    
