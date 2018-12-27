# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 12:00:51 2017

@author: hbraun
"""

from vampyre.trans.base import BaseLinTrans
import numpy as np
from scipy.fftpack import fft2, ifft2


class Fourier2LT(BaseLinTrans):
    """Linear transform defined by a 2-D Fourier transform.
    
    The class defines a linear transform :math:`z_1 = Fz_0` where :math:`F`
    executes a 2-D FFT on each column of Z_0. Each column must be able to be 
    reshaped to the instance's fft_shape.
    
    :note: The current code has not been tested for inputs of more than 2 dimensions.
    
    :param fft_shape: (height,width) of input signal
    :param fft_axes: (axis1, axis2) axes along which to perform the Fourier transform.
    
    """
    def __init__(self, in_shape, dtype=np.float64,name=None, 
             fft_shape=None, fft_axes=(-2, -1)):
        
        self.fft_shape = fft_shape
        self.fft_axes = fft_axes
        
        shape0 = in_shape
        shape1 = shape0
        shape1[fft_axes] = self.fft_shape
        
        BaseLinTrans.__init__(self, shape0, shape1, dtype, dtype, 
              svd_avail=True,name=name)
    
    def dot(self, x):
        y = fft2(x, norm=self.norm, axes=self.fft_axes)
        return np.array(y, dtype=self.dtype1)
        
    def dotH(self,y):
        x = ifft2(y, axes=self.fft_axes)
        x = np.prod(self.fft_shape) * x
        
        #if forward operation was zero-padded, crop the output back down to 
        #size. if it was cropped, zero-pad it back to size.
        slc = slice(None) * len(self.shape0)
        pad = np.zeros((2,len(self.shape0)))
        for i_ax in self.ft_axes:
            if self.shape1[i_ax] > self.shape0[i_ax]:
                slc[i_ax] = slice(self.shape0)
            elif self.shape1[i_ax] < self.shape0[i_ax]:
                pad[1,i_ax] = self.shape0[i_ax] - self.shape1[i_ax]
        x = np.pad(x[slc], pad)
                
                
        
        return np.array(x, dtype=self.dtype0)