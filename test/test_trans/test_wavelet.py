"""
test_wavelet.py:  Test suite for the wavelet module
"""
from __future__ import print_function, division

# Removes the warning that "Tensorflow library was not compiled to use SSE..."
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import unittest
import numpy as np

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp

def wavelet2d_test(nrow=256,ncol=256,verbose=False,tol=1e-8):
    """
    Unit test for the Wavelet2DLT class.
    
    The test creates a random Gaussian image and verifies that inner product
    and reconstruction
    
    :param nrow:  number of rows in the image
    :param ncol:  number of columns in the image
    :param tol:  tolerance above which test fails
    :param Booolean verbose:  Boolean flag indicating if the
       results are to be displayed
    """
    
    # Create a wavelet transform class
    wv = vp.trans.Wavelet2DLT(nrow=nrow,ncol=ncol)
    
    # Create a random input and output
    z0 = np.random.normal(0,1,(nrow,ncol))
    u1 = np.random.normal(0,1,(nrow,ncol))
    
    # Validate reconstruction
    z1 = wv.dot(z0)
    z2 = wv.dotH(z1)
    recon_err = np.sum((z2-z0)**2)
    fail = (recon_err > tol)
    if verbose or fail:
        print("Reconstruction error {0:12.4e}".format(recon_err))
    if fail:
        raise vp.common.TestException("Reconstruction error exceeded tolerance")
            
    
        
    # Inner product test
    u0 = wv.dotH(u1)
    ip0 = np.sum(z0*u0)
    ip1 = np.sum(z1*u1)
    ip_err = np.abs(ip0-ip1)
    fail = (ip_err > tol)
    if verbose or fail:
        print("Inner product error {0:12.4e}".format(ip_err))
    if fail:
        raise vp.common.TestException("Inner products do not match within tolerance")
            

class TestCases(unittest.TestCase):
    def test_wavelet2d(self):
        """
        Run the conv2d test.
        
        Note that on Windows 10 with tensorflow 1.0, there is a long warning 
        that can be ignored.  This will be fixed in the next TF release.
        """        
        wavelet2d_test(verbose=False)
            
        
if __name__ == '__main__':    
    unittest.main()
    
