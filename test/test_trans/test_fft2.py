"""
test_fft2.py: test suite for the fft2 transform module
"""

from __future__ import print_function, division

import unittest
import numpy as np

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp

def fft2_round_trip_is_consistent(nrow=256,ncol=256, err_tol = 1E-12):
    fft_size = (nrow, ncol)
    raise NotImplementedError
    
class TestCases(unittest.TestCase):
    def test_fft2_round_trip_consistency(self):
        """
        Run the fft2 test case.
        """
        fft_size = (256,256)
        err_tol = (1E-12)
        
        x = np.random.normal(size=fft_size)
        fft_op = vp.trans.Fourier2DLT(fft_size,fft_shape=fft_size)
        scale_factor = 1/np.prod(fft_size)
        
        x_fft = fft_op.dot(x)
        x_roundtrip = scale_factor * fft_op.dotH(x_fft)

        
        mse = np.sum((x - x_roundtrip)**2)
        self.assertLess(mse,err_tol)
        
        

if __name__ == '__main__':    
    unittest.main()