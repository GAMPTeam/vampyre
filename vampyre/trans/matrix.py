import numpy as np
from vampyre.trans.base import LinTrans
from vampyre.common.utils import repeat_axes

class MatrixLT(LinTrans):
    """
    Linear transform defined by a matrix
    
    The class defines a linear transform :math:`z_1 = Az_0` where :math:`A`
    is represented as a matrix.
    
    
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
        Ashape = A.shape
        shape1 = np.array(shape0)
        shape1[0] = Ashape[0]
        self.shape1 = tuple(shape1)
        
        # Set SVD terms to not computed
        self.svd_computed = False
        
    def dot(self,z0):
        """
        Compute matrix multiply :math:`A(z0)`
        """
        return self.A.dot(z0)
        
    def dotH(self,z0):
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
        
        # Compute the shape 
        self.sshape = np.array(self.shape0)
        self.sshape[0] = len(s)
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
        return self.U.conj.T.dot(z1)
        
    
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
        
        :returns: :code:`s, sshape, srep_axes` where 
           :code:`s` is the singular values, :code:`sshape` is the shape
           of the parameters of the 
        """
        self._comp_svd()
        return self.s, self.sshape, self.srep_axes
        

def matrix_test(Ashape=(50,100),shape0=(100,10), tol=1e-8, verbose=False):
    """
    Unit test for the :class:`MatrixLT` class.
    
    :param Ashape:  Shape of matrix
    :param shape0:  Shape of input
    :param Boolean verbose:  Print results of test.
    :param tol:  Tolerance for passing test
    
    :return: :code:`fail` indicating if test passed.
    """
    # Generate random matrix, input and output
    A = np.random.uniform(0,1,Ashape)
    z0 = np.random.uniform(0,1,shape0)
    z1 = A.dot(z0)

    # Create corresponding matrix operator
    Aop = MatrixLT(A,shape0)

    # Make sure matrix multiplication works
    s, sshape, srep_axes = Aop.get_svd_diag()        
    q0 = Aop.VsvdH(z0)
    srep = repeat_axes(s,sshape,srep_axes,rep=False)
    q1 = srep*q0
    z1est = Aop.Usvd(q1)
    
    err = np.linalg.norm(z1-z1est)
    fail = (err > tol)
    if verbose:
        if fail:
            print("Error: {0:12.4e} fail".format(err))
        else:
            print("Error: {0:12.4e} pass".format(err))
    
    return fail        
    
    