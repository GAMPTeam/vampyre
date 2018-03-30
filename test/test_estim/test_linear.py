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

        
def lin_test(zshape=(500,10),Ashape=(1000,500),verbose=False,tol=0.1):
    """
    Unit test for the linear estimator class
    
    The test is performed by generating random data 
    
    :math:`y=Az+w,  z \\sim {\\mathcal N}(r, \\tau_r I), 
       w \\sim {\\mathcal N}(0, \\tau_w I)`
       
    Then the method estimates :math:`z` from :math:`y` 
    and compares the expected and measured errors.
    
    :param zshape:  shape of :math:`z`
    :param Ashape:  shape of :A:`z`.  This must be consistent with 
       :code:`zshape`.
    :param Boolenan verbose:  print results
    :param tol:  error tolerance above which test is considered
       to fail.    
    """            
     
    # Generate random parameters
    rvar = 10**(np.random.uniform(-1,1,1))[0]
    wvar = 10**(np.random.uniform(-1,1,1))[0]    
        
    # Generate random matrix
    A = np.random.normal(0,1,Ashape)/np.sqrt(Ashape[1])
    Aop = vp.trans.MatrixLT(A, zshape)
    yshape = Aop.shape1
    
    # Add noise on input and output
    r = np.random.normal(0,1,zshape) 
    z = r + np.random.normal(0,np.sqrt(rvar),zshape)
    y = A.dot(z) + np.random.normal(0,np.sqrt(wvar),yshape)
    
    # Construct the linear estimator
    est = vp.estim.LinEst(Aop,y,wvar,var_axes='all')
    
    # Perform the initial estimate.  This is just run to make sure it
    # doesn't crash
    zhat, zhatvar, cost = est.est_init(return_cost=True)
    if (zhat.shape != r.shape) or (zhatvar.shape != wvar):
        raise vp.common.TestException(\
           "est_init does not produce the correct shape")            
    
    # Posterior estimate
    zhat, zhatvar, cost = est.est(r,rvar,return_cost=True)
    zerr = np.mean(np.abs(z-zhat)**2)
    fail = (np.abs(zerr-zhatvar) > tol*np.abs(zhatvar))
    if verbose or fail:
        print("\nPosterior:    True: {0:f} Est:{1:f}".format(zerr,zhatvar))
    if fail:
       raise vp.common.TestException("Posterior estimate Gaussian error "+ 
          " does not match predicted value")      
    

class TestCases(unittest.TestCase):
    def test_linear(self):
       lin_test(zshape=(500,10),Ashape=(1000,500))
       lin_test(zshape=(500,),Ashape=(1000,500),tol=0.5)
       lin_test(zshape=(500,10),Ashape=(250,500))
        
if __name__ == '__main__':    
    unittest.main()
    
