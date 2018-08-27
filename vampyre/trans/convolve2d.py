"""
convolve2d.py:  Linear transforms for 2D convolution
"""
from __future__ import division

import numpy as np

# Import individual classes from same modules in the same package
from vampyre.trans.base import BaseLinTrans

class Convolve2DLT(BaseLinTrans):
    def __init__(self,shape,kernel,is_complex=False,im_axes=(0,1)):
        """
        2D convolution linear transform
        
        :param shape:  image tensor shape 
        :param kernel:  image domain kernel of shape :code:`(nrowk,ncolk)`.
            Currently, :code:`nrowk` and :code:`ncolk` must be odd length
            so that the kernel can be re-centered to `(0,0)`.
        :param is_complex:  output is complex.
        :param im_axes:  the axes of the image to be convolved
            The other axes are the channels on which the kernel will be applied
        
        :note:  The transform currently only supports circular convolutions
        """
        
        # Save parameters
        self.shape0 = shape
        self.shape1 = shape        
        self.svd_avail = True
        self.is_complex = is_complex
        self.im_axes = im_axes
        
        # Compute the axes of the channels of the image
        ndim = len(shape)
        self.chan_axes = tuple((i for i in range(ndim) if (i not in im_axes)))
        
        # Get dimensions                    
        nrowk, ncolk = kernel.shape
        nrow = shape[im_axes[0]]
        ncol = shape[im_axes[1]]
        if (nrowk % 2 == 0) or (ncolk % 2 == 0):
            raise ValueError("Kernel dimensions must be odd")
        if (nrowk > nrow) or (ncolk > ncol):
            raise ValueError("Kernel must be smaller than image")

        # Compute FFT scale factor so the FFT and IFFT are unitary        
        self.scale = np.sqrt(nrow*ncol)        

        # Zero pad the kernel so that it is the same shape as the input
        # Also, the kernel is centered.                            
        kernel1 = np.zeros((nrow,ncol))
        kernel1[:nrowk//2+1,:ncolk//2+1] = kernel[nrowk//2:, ncolk//2:]
        kernel1[:nrowk//2+1,-ncolk//2+1:] = kernel[nrowk//2:, :ncolk//2]
        kernel1[-nrowk//2+1:,-ncolk//2+1:] = kernel[:nrowk//2, :ncolk//2]
        kernel1[-nrowk//2+1:,:ncolk//2+1] = kernel[:nrowk//2, ncolk//2:]
                
        # Pre-compute the FFT of the kernel
        self.kernel_fft = np.fft.fft2(kernel1)

        # Repeat the kernel over all dimensions
        ker_shape = np.ones(ndim,dtype=int)
        ker_shape[self.im_axes[0]] = nrow
        ker_shape[self.im_axes[1]] = ncol
        self.kernel_fft = self.kernel_fft.reshape(ker_shape)

        self.srep_axes = self.chan_axes
        
        # Superclass constructor
        dtype = np.float
        BaseLinTrans.__init__(self, shape, shape, dtype, dtype,\
                              svd_avail=True,name='Convolve2D')

        
    def Usvd(self,q1):
        """
        Multiplication by SVD term :math:`U` 
        """    
        z1 = np.fft.ifft2(q1,axes=self.im_axes)*self.scale
        if not self.is_complex:
            z1 = np.real(z1)
        return z1

    def UsvdH(self,z1):
        """
        Multiplication by SVD term :math:`U^*` 
        """    
        return np.fft.fft2(z1,axes=self.im_axes)/self.scale
        
    
    def Vsvd(self,q0):
        """
        Multiplication by SVD term :math:`V` 
        """
        z0 = np.fft.ifft2(q0,axes=self.im_axes)*self.scale
        if not self.is_complex:
            z0 = np.real(z0)
        return z0
        
    def VsvdH(self,z0):
        """
        Multiplication by SVD term :math:`V^*` 
        """    
        return np.fft.fft2(z0,axes=self.im_axes)/self.scale
            
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
        nrow = self.shape0[self.im_axes[0]]
        ncol = self.shape0[self.im_axes[1]]
        s = np.reshape(self.kernel_fft, (nrow,ncol))
        sshape = self.shape0
        srep_axes = self.chan_axes
        return s, sshape, srep_axes
        
    def dot(self,z0):
        """
        Forward multiplication
        """
        z1 = self.Usvd(self.kernel_fft*self.VsvdH(z0))
        if not self.is_complex:
            z1 = np.real(z1) 
        return z1

    def dotH(self,z1):
        """
        Adjoint multiplication
        """
        z0 = self.Vsvd(np.conj(self.kernel_fft)*self.UsvdH(z1))
        if not self.is_complex:
            z0 = np.real(z0)         
        return z0

    