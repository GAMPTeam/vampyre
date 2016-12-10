class LinTrans(object):
    """
    Linear transform base class
    
    The class represents deterministic lienar operations :math:`z_1=Az_0`.
    
    **SVD decomposition**
    
    Some estimators require an SVD-like decomposition.  The SVD decomposition
    is assumed to be of the form :math:`A = U\\mathrm{diag}(s)V^*` meaning that
    we can write the transform :math:`z_1 = Az_0` as
    
    :math:`q_0 = V^*z_0, q_1=\\mathrm{diag}(s)q_0, z_1=Uq_1`.
    
    The linear transforms :math:`U` and :math:`V` are implemented by
    methods that derive from this class.   The attribute :code:`svd_avail`
    indicates if the linear transform class supports an SVD.      
    
    :note:  We say the decomposition is *SVD-like* since we do not require
        that :math:`s` is real and positive.  It may have complex values.
        Hence, the singular values are given by :math:`|s|`.
    """    
    def __init__(self):
        self.svd_avail = False
        
    def dot(self,z0):
        """
        Compute matrix multiply :math:`A(z0)`
        """
        raise NotImplementedError()
        
    def dotH(self,z0):
        """
        Compute conjugate transpose multiplication :math:`A^*(z1)`
        """
        raise NotImplementedError()       
    
    def Usvd(self,q1):
        """
        Multiplication by SVD term :math:`U` 
        """        
        raise NotImplementedError()
        
    def UsvdH(self,z1):
        """
        Multiplication by SVD term :math:`U^*` 
        """            
        raise NotImplementedError()
            
    def Vsvd(self,q0):
        """
        Multiplication by SVD term :math:`V` 
        """        
        raise NotImplementedError()
        
    def VsvdH(self,z0):
        """
        Multiplication by SVD term :math:`V^*` 
        """    
        raise NotImplementedError()     
    
    def get_svd_diag(self):
        """
        Gets parameters of the SVD diagonal multiplication.
        
        It is assumed that the transformed output
        :code:`q0 = self.VsvdH(z0)` is represented as an
        :class:`numpy.ndarray`.  This method provides variables
        to perform the diagonal multiplication in this space.
        With the parameters the forward multiplication :code:`z1=A.dot(z0)` should 
        be equivalent to::

            import vampyre as vp
            
            Aop = ... # LinTrans object with SVD enabled
            q0   = Aop.VsvdH(z0)
            srep = vp.common.repeat_axes(s,sshape,srep_axes,rep=False)
            q1   = srep*q0
            z1   = Aop.Usvd(q1)
        
        :returns: :code:`s, sshape, srep_axes` where 
           :code:`s` is the singular values, :code:`sshape` is the shape
           of the parameters in the transformed space and 
           :code:`srep_axes` are the list of axes on which the terms
           :code:`s` are to be repeated.
        """
        raise NotImplementedError()
        
        
