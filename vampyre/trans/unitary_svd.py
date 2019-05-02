#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 09:56:39 2018

@author: hbraun
"""
import vampyre.common as common
import numpy as np


class UnitarySvdMixin(object):
    """
    Add SVD calculations to an object, for the special case of 
    a unitary matrix.  In this case, the SVD is given as 
    
    :math: `M = U S V = M (\alpha I) I`
    
    and the SVD can be easily applied.
    
    To use, inherit the OrthogonalSvdMixin class and supply the following 
    properties and methods: 
        dot() 
        dotH()
        shape0
        

    """
    
    def __init__(self, ):
        self.svd_avail = True
    
    def is_orthogonal(self, err_tol = 1E-12):
        """
        Return True if the dot() and dotH() operations implement an orthogonal 
        transformation, and False otherwise.
        """
        raise NotImplementedError
    
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
        s = self.alpha
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
