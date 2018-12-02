"""
tflintrans.py:  Linear transforms based on Tensorflow ops
"""
from __future__ import division

import pywt
import numpy as np

# Import other subpackages in vampyre
import vampyre.common as common

# Import individual classes from same modules in the same package
from vampyre.trans.base import BaseLinTrans


class Wavelet2DLT(BaseLinTrans):
    """
    Linear transform class based on a 2D wavelet

    :param nrow:  number of rows in the image
    :param ncol:  number of columns in the image
    :param wavelet:  wavelet type (see `pywt` package for a full description)
    :param level:  number of wavelet levels
    :param fwd_mode:  `recon` indicates that `dot()` operation is the
       reconstruction and `dotH()` is the analysis.  `analysis` is the reverse.
    """
    def __init__(self,nrow=256,ncol=256,wavelet='db4',level=3,fwd_mode='recon',\
        dtype=np.float64,name=None):

        # Save parameters
        self.wavelet = wavelet
        self.level = level
        shape0 = (nrow,ncol)
        shape1 = (nrow,ncol)
        dtype0 = dtype
        dtype1 = dtype

        if pywt.Wavelet(wavelet).orthogonal:
            svd_avail = True #SVD calculation assumes an orthogonal wavelet
        else:
            svd_avail = False
        BaseLinTrans.__init__(self, shape0, shape1, dtype0, dtype1,\
           svd_avail=svd_avail,name=name)


        # Set the mode to periodic to make the wavelet orthogonal
        self.mode = 'periodization'

        # Send a zero image to get the coefficient slices
        im = np.zeros((nrow,ncol))
        coeffs = pywt.wavedec2(im, wavelet=self.wavelet, level=self.level, \
            mode=self.mode)
        _, self.coeff_slices = pywt.coeffs_to_array(coeffs)


        # Confirm that fwd_mode is valid
        if (fwd_mode != 'recon') and (fwd_mode != 'analysis'):
            raise common.VpException('fwd_mode must be recon or analysis')
        self.fwd_mode = fwd_mode

    def dot(self,z0):
        """
        Forward multiplication
        """
        if (self.fwd_mode == 'recon'):
            z1 = self.recon(z0)
        else:
            z1 = self.analysis(z0)
        return z1

    def dotH(self,z1):
        """
        Reverse / adjoint multiplication
        """
        if (self.fwd_mode == 'recon'):
            z0 = self.analysis(z1)
        else:
            z0 = self.recon(z1)
        return z0

    def analysis(self,z0):
        """
        Analysis:  image -> coefficients
        """
        coeffs = pywt.wavedec2(z0, wavelet=self.wavelet, level=self.level, \
            mode=self.mode)
        z1, _ = pywt.coeffs_to_array(coeffs)
        return z1

    def recon(self,z1):
        """
        Wavelet reconstruction:  coefficients -> image
        """
        coeffs = pywt.array_to_coeffs(z1, self.coeff_slices, \
            output_format='wavedec2')
        z0 = pywt.waverec2(coeffs, wavelet=self.wavelet, mode=self.mode)
        return z0

    def Usvd(self,q1):
        """
        Multiplication by SVD term :math:`U`
        """
        return self.dot(q1)

    def UsvdH(self,z1):
        """
        Multiplication by SVD term :math:`U^*`
        """
        return self.dotH(z1)


    def Vsvd(self,q0):
        """
        Multiplication by SVD term :math:`V`
        """
        return q0

    def VsvdH(self,z0):
        """
        Multiplication by SVD term :math:`V^*`
        """
        return z0

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
        s = 1
        sshape = self.shape0
        srep_axes = (0,1)
        return s, sshape, srep_axes

    def svd_dot(self,s1,q0):
        """
        Performs diagonal matrix multiplication.

        Implements :math:`q_1 = \\mathrm{diag}(s_1) q_0`.

        :param s1: diagonal parameters
        :param q0: input to the diagonal multiplication
        :returns: :code:`q1` diagonal multiplication output
        """
        srep = common.repeat_axes(s1,self.shape0, (0,1),rep=False)
        q1 = srep*q0
        return q1

    def svd_dotH(self,s1,q1):
        """
        Performs diagonal matrix multiplication conjugate

        Implements :math:`q_0 = \\mathrm{diag}(s_1)^* q_1`.

        :param s1: diagonal parameters
        :param q1: input to the diagonal multiplication
        :returns: :code:`q0` diagonal multiplication output
        """
        srep = common.repeat_axes(np.conj(s1),self.shape0,(0,1),rep=False)
        q0 = srep*q1
        return q0
