class LinTrans(object):
    """
    Linear transform base class
    
    The class provides methods for linear operations :math:`z_1=Az_0`.
    
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
        
    def dotH(self,z1):
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
        Gets diagonal parameters of the SVD diagonal multiplication.
        
        The method returns a set of diagonal parameters, :param:`s`, 
        in the SVD-like decomposition.  With the parameters the 
        forward multiplication :code:`z1=A.dot(z0)` should be equivalent to::

            
            Aop = ... # LinTrans object with SVD enabled
            s  = Aop.get_svd_diag()[0]
            q0 = Aop.VsvdH(z0)
            q1 = Aop.svd_dot(s, q0)
            z1 = Aop.Usvd(q1)
            
        One can also compute any function of the matrix.  Suppose
        :math:`f` is a any continuous function such that :math:`f(0)=0`.
        Then, the transform :math:`Uf(S)V^*z_0` is equivalent to::
        
            s  = Aop.get_svd_diag()[0]
            q0 = Aop.VsvdH(z0)
            q1 = Aop.svd_dot(f(s), q0)
            z1 = Aop.Usvd(q1)
        
        :returns: :code:`s,sshape,srep_axes`, the diagonal parameters 
            :code:`s`, the shape in the transformed domain :code:`sshape`,
            and the axes on which the diagonal parameters are to be 
            repeated, :code:`srep_axes`  
        """
        raise NotImplementedError()
        
    def svd_dot(self,s1,q0):
        """
        Performs diagonal matrix multiplication. 
        
        Implements :math:`q_1 = \\mathrm{diag}(s_1) q_0`.
        
        :param s1: diagonal parameters
        :param q0: input to the diagonal multiplication
        :returns: :code:`q1` diagonal multiplication output
        """
        raise NotImplementedError()
        
    def svd_dotH(self,s1,q1):
        """
        Performs diagonal matrix multiplication conjugate
        
        Implements :math:`q_0 = \\mathrm{diag}(s_1)^* q_1`.
        
        :param s1: diagonal parameters
        :param q1: input to the diagonal multiplication
        :returns: :code:`q0` diagonal multiplication output
        """
        raise NotImplementedError()
        

            
        
