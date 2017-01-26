"""
matrix.py:  Linear transforms based on a matrix
"""
from __future__ import division

import numpy as np
from vampyre.trans.base import LinTrans
from vampyre.common.utils import repeat_axes
from vampyre.common.utils import TestException

class MatrixLT(LinTrans):
    """
    Linear transform defined by a matrix
    
    The class defines a linear transform :math:`z_1 = Az_0` where :math:`A`
    is represented as a :class:`numpy.ndarray`.
    
    :note: The current code assumes that :param:`A` is either a 1D or 2D
       array.  For higher dimensions, it may be good to develop an alternate
       tensor class using the :func:`numpy.ndarray.tensordot` method.
        
    :param A:  matrix 
    :param shape0:  input shape (The output shape is computed from this)
    """
    def __init__(self, A, shape0):
        LinTrans.__init__(self)
        self.A = A
        if np.isscalar(shape0):
            shape0 = (shape0,)
        self.shape0 = shape0
        
        # Compute the output shape
        # Note that A.dot(x) operates on the second to last axis of x
        Ashape = A.shape
        shape1 = np.array(shape0)
        if len(shape0) == 1:
            self.aaxis = 0
        else:
            self.aaxis = len(shape0)-2
        shape1[self.aaxis] = Ashape[0]
        self.shape1 = tuple(shape1)
        
        # Set SVD terms to not computed
        self.svd_computed = False
        self.svd_avail = True
        
    def dot(self,z0):
        """
        Compute matrix multiply :math:`A(z0)`
        """
        return self.A.dot(z0)
        
    def dotH(self,z1):
        """
        Compute conjugate transpose multiplication:math:`A^*(z1)`
        """
        return self.A.conj().T.dot(z1)
        

    def _comp_svd(self):
        """
        Compute the SVD terms, if necessary
        
        If the SVD is already computed, simply return
        """      
        # Return if SVD is already computed
        if self.svd_computed:
            return
        
        # Compute SVD.  Note that linalg.svd returns V, not its
        # conjugate transpose as is usual.
        U,s,V = np.linalg.svd(self.A, full_matrices=False)
        self.U = U
        self.s = s
        self.V = V.conj().T
        self.svd_computed = True
                
        # Compute the shape of the transformed space
        self.sshape = np.array(self.shape0)
        self.sshape[self.aaxis] = len(s)
        self.sshape = tuple(self.sshape)
        
        # Compute the axes on which the diagonal multiplication
        # is to be repeated.  This is all but axis 0
        ndim = len(self.sshape)
        self.srep_axes = tuple(range(1,ndim))        
                
    def Usvd(self,q1):
        """
        Multiplication by SVD term :math:`U` 
        """
        self._comp_svd()
        return self.U.dot(q1)
        
    def UsvdH(self,z1):
        """
        Multiplication by SVD term :math:`U^*` 
        """    
        self._comp_svd()
        return self.U.conj().T.dot(z1)
        
    
    def Vsvd(self,q0):
        """
        Multiplication by SVD term :math:`V` 
        """
        self._comp_svd()
        return self.V.dot(q0)
        
    def VsvdH(self,z0):
        """
        Multiplication by SVD term :math:`V^*` 
        """    
        self._comp_svd()
        return self.V.conj().T.dot(z0)
            
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
        self._comp_svd()
        return self.s, self.sshape, self.srep_axes
        
    def svd_dot(self,s1,q0):
        """
        Performs diagonal matrix multiplication. 
        
        Implements :math:`q_1 = \\mathrm{diag}(s_1) q_0`.
        
        :param s1: diagonal parameters
        :param q0: input to the diagonal multiplication
        :returns: :code:`q1` diagonal multiplication output
        """
        srep = repeat_axes(s1,self.sshape,self.srep_axes,rep=False)
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
        srep = repeat_axes(np.conj(s1),self.sshape,self.srep_axes,rep=False)
        q0 = srep*q1
        return q0
                    
        
def matrix_test(Ashape=(50,100),shape0=(100,10), tol=1e-8, verbose=False,\
                raise_exception=True):
    """
    Unit test for the :class:`MatrixLT` class.
    
    :param Ashape:  Shape of matrix
    :param shape0:  Shape of input
    :param Boolean verbose:  Print results of test.
    :param tol:  Tolerance for passing test
    :param Boolean raise_exception:  Raises an error on test failure.  This 
        can be caught in the unit test dispatcher.
    """
    # Generate random matrix, input and output
    A = np.random.uniform(0,1,Ashape)
    z0 = np.random.uniform(0,1,shape0)
    z1 = A.dot(z0)

    # Create corresponding matrix operator
    Aop = MatrixLT(A,shape0)

    # Perform the multiplication via SVD
    s = Aop.get_svd_diag()[0]     
    q0 = Aop.VsvdH(z0)
    q1 = Aop.svd_dot(s,q0)
    z1est = Aop.Usvd(q1)
    
    # Test fails if SVD method does not match direct method
    err = np.linalg.norm(z1-z1est)
    if verbose:
        print("Error: {0:12.4e}".format(err))        
    if (err > tol) and raise_exception:
        raise TestException("SVD method for performing multiplication"\
            +"does not match direct multiplication.  "\
            +"err={0:12.4e}".format(err))
        
    