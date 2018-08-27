"""
test_convolve2d.py:  Test for the :class:`Convolve2DLT` class.
"""
from __future__ import print_function, division

import unittest
import numpy as np
from scipy import signal

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp

def test_convolve2d(shape=(128,128,3), tol=1e-8, verbose=False):
    """
    Unit test for the :class:`MatrixLT` class.
    
    :param shape:  Input and output shape
    """
    # Generate random matrix, input and output
    x = np.random.normal(0,1,shape)
    
    # Generate random kernel
    shapek = (5,5)
    kernel = np.random.normal(0,1,shapek)
    
    # Convolve using scipy
    # The signal.convolve2d only does one channel at a time
    ndim = len(shape)
    if (ndim==2):    
        y = signal.convolve2d(x,kernel,mode='same',boundary='wrap')
    elif (ndim==3):
        y = np.zeros(shape)
        nchan = shape[2]
        for i in range(nchan):
           y[:,:,i] = signal.convolve2d(x[:,:,i],kernel,mode='same',boundary='wrap')
    else:
        raise ValueError('shape must be 2 or 3 dimensional')
    
    
    # Convolve using LT
    conv_trans = vp.trans.Convolve2DLT(shape, kernel)
    yhat0 = conv_trans.dot(x)
    
    # Convolve using SVD
    s, sshape, srep_axes  = conv_trans.get_svd_diag()
    q0 = conv_trans.VsvdH(x)
    s1 = vp.common.utils.repeat_axes(s,sshape,srep_axes,rep=False)
    q1 = s1*q0
    yhat1 = conv_trans.Usvd(q1)
    
    # Test fails if the convolution does not match
    err0 = np.linalg.norm(y-yhat0)
    err1 = np.linalg.norm(y-yhat1)
    fail = (err0 > tol) or (err1 > tol)
    if verbose or fail:
        print("Error: Direct %12.4e SVD %12.4e" % (err0, err1))        
    if fail:
        raise vp.common.TestException("Convolution does not match expected output "\
            +"err={0:12.4e}, {1:12.4e}".format(err0, err1))
            
          
class TestCases(unittest.TestCase):
    def test_convolve2d(self):
        """
        Tests the matrix.
        """        
        verbose = False
        test_convolve2d(shape=(128,128),verbose=verbose)
        test_convolve2d(shape=(128,128,3),verbose=verbose)
            
        
if __name__ == '__main__':    
    unittest.main()
    
