# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 12:00:51 2017

@author: hbraun
"""

from vampyre.trans.base import BaseLinTrans
from vampyre.trans.unitary_svd import UnitarySvdMixin
import numpy as np
from scipy.fftpack import fft2, ifft2


class Fourier2DLT(UnitarySvdMixin, BaseLinTrans):
    """Linear transform defined by a 2-D Fourier transform.
    
    The class defines a linear transform :math:`z_1 = Fz_0` where :math:`F`
    executes a 2-D FFT on each column of Z_0. Each column must be able to be 
    reshaped to the instance's fft_shape.
    
    :note: The current code has not been tested for inputs of more than 2 
    dimensions.
    
    :param in_shape: (height,width) of input signal
    :param fft_axes: (axis1, axis2) axes along which to perform the Fourier 
        transform.
    :param fft_shape: (shape_axis1, shape_axis2) size of the FFT for each axis.
        If the size is 
    
    """
    def __init__(self, in_shape, dtype=np.complex128,name=None, 
             fft_shape=None, fft_axes=(-2, -1)):
        
        
        self.fft_axes = fft_axes
        
        shape0 = np.asarray(in_shape)
        shape1 = shape0
        if fft_shape is None:
            fft_shape = [in_shape[i] for i in self.fft_axes]
        else: 
            shape1[(fft_axes,)] = fft_shape
        self.fft_shape = fft_shape
        self._scale_factor = (np.prod(self.fft_shape))
        
        BaseLinTrans.__init__(self, shape0, shape1, dtype, dtype, 
              svd_avail=True,name=name)
    
    def dot(self, x):
        y = fft2(x, shape=self.fft_shape, axes=self.fft_axes)
        return np.array(y, dtype=self.dtype1)
        
    def dotH(self,y):
        x = ifft2(y, axes=self.fft_axes)
        x = np.prod(self.fft_shape) * x
        
        #if forward operation was zero-padded, crop the output back down to 
        #size. if it was cropped, zero-pad it back to size.
        if not np.all(self.shape0 == self.shape1):
            slc = slice(None) * len(self.shape0)
            pad = np.zeros((2,len(self.shape0)))
            for i_ax in self.ft_axes:
                if self.shape1[i_ax] > self.shape0[i_ax]:
                    slc[i_ax] = slice(self.shape0)
                elif self.shape1[i_ax] < self.shape0[i_ax]:
                    pad[1,i_ax] = self.shape0[i_ax] - self.shape1[i_ax]
            x = np.pad(x[slc], pad)
                
        return np.array(x, dtype=self.dtype0)
    
    def Usvd(self,q1):
        """
        Multiplication by SVD term :math:`U` 
        """        
        return self._scale_factor**-1 * UnitarySvdMixin.Usvd(self,q1)
    
    def UsvdH(self,z1):
        """
        Multiplication by SVD term :math:`U^*` 
        """            
        return self._scale_factor**-1 * UnitarySvdMixin.Usvd(self,z1)

    # Vsvd() and VsvdH() are the same as for a unitary matrix and we stick with
    # the versions provided by UnitarySvdMixin
    
    def get_svd_diag(self):
        """
        Gets parameters of the SVD diagonal multiplication.

        See :func:`vampyre.trans.base.LinTrans.get_svd_diag()` for
        more information.

        :returns: :code:`s,sshape,srep_axes`, the diagonal parameters
            :code:`s`, the shape in the transformed domain :code:`sshape`,
            and the axes on which the diagonal parameters are to be
            repeated, :code:`srep_axes`
        """
        (s, sshape, srep_axes) = UnitarySvdMixin.get_svd_diag(self)
        s = self._scale_factor * s
        return s, sshape, srep_axes