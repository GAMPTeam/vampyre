"""
tflintrans.py:  Linear transforms based on Tensorflow ops
"""
from __future__ import division

import pywt
import numpy as np

# Import other subpackages in vampyre
import vampyre.common as common

# Import individual classes from same modules in the same package
from vampyre.trans.base import LinTrans


class Wavelet2DLT(LinTrans):
    """
    Linear transform class based on a 2D wavelet
    
    :param nrow:  number of rows in the image
    :param ncol:  number of columns in the image
    :param wavelet:  wavelet type (see `pywt` package for a full description)
    """    
    def __init__(self,nrow=256,ncol=256,wavelet='db4'):
        # Save parameters
        self.wavelet = wavelet
        self.shape0 = (nrow,ncol)
        self.shape1 = (nrow,ncol)
        
        # Create a wavelet packet that will be used for the conjugate operation
        im = np.zeros((nrow,ncol))
        self.wp_tr = pywt.WaveletPacket2D(data=im, wavelet=self.wavelet, \
            mode='symmetric')
    
    def dot(self,z0):
        """
        Forward multiplication
        """
        wp = pywt.WaveletPacket2D(data=z0, wavelet=self.wavelet, \
            mode='symmetric')
        z1 = wp.data    
        return z1
        
    def dotH(self,z1):
        """
        Adjoint multiplication.  This is computed as a gradient of z_op
        """
        self.wp_tr.data = z1
        z0 = self.wp_tr.reconstruct()
        return z0
        

nrow = 256
ncol = 256

wv = Wavelet2DLT(nrow=nrow,ncol=ncol)
z0 = np.random.normal(0,1,(nrow,ncol))
z1 = wv.dot(z0)
z2 = wv.dotH(z1)

        

    