# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 12:00:51 2017

@author: hbraun
"""

from vampyre.trans.base import LinTrans
import numpy as np


class Fourier2LT(LinTrans):
    """Linear transform defined by a 2-D Fourier transform.
    
    The class defines a linear transform :math:`z_1 = Fz_0` where :math:`F`
    executes a 2-D FFT on each column of Z_0. Each column must be able to be 
    reshaped to the instance's fftShape.
    
    :note: The current code has not been tested for inputs of more than 2 dimensions.
    
    :param fftShape: (height,width) of input signal
    :param norm: 'ortho' for unitary FFT, None for un-normalized
    
    """
    def __init__(self, fftShape, norm='ortho'):
        LinTrans.__init__(self)
        
        #constants
        self._FT_AXES = (0,1)
        
        #set parameters
        self.fftShape = fftShape
        self.norm = norm
        
    def _getNormCoeff(self):
        coeff = np.prod(self.fftShape)
        return coeff
    
    def dot(self, x):
        ft_input = np.reshape(x, self.fftShape + x.shape[1:])
        ft_out = np.fft.fft2(ft_input, norm=self.norm, axes=self._FT_AXES)
        y = np.reshape(ft_out, np.shape(x))
        return y
        
    def dotH(self,y):
        ift_input = np.reshape(y, self.fftShape + y.shape[1:])
        ift_out = np.fft.ifft2(ift_input, norm=self.norm, axes=self._FT_AXES)
        if self.norm is not 'ortho': #normalize if needed
            print('re-normalizing!')
            ift_out = self._getNormCoeff() * ift_out
        x = np.reshape(ift_out, np.shape(y))
        return x