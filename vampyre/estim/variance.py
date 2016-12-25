# -*- coding: utf-8 -*-
"""
variance.py:  Basic classes for variance handling
"""
import numpy as np

# Import other sub-packages
import vampyre.common as common

class VarHdl(object):
    """
    Base class for an estimation variance handler
    """
    def __init__(self):
        pass
    
    def msg_sub(self,z,zvar,r0,rvar0,r1_prev=[],rvar1_prev=[]):
        """
        Variance subtraction for message passing
        
        Suppose that a factor node computes a 
        mean and variance :math:`(z,\\tau_z)` based on an 
        incoming Gaussian message with parameters :math:`(r_0,\\tau_0)`.  
        This method then computes the outgoing message parameters 
        :math:`(r_1,\\tau_1)` from the factor node back to the variable node 
        using the subtraction:
        
        :math:`\\tau_1 = \\tau_0\\tau_z / (\\tau_0 - \\tau_z)`
        
        :math:`r_1 = \\tau_1( z/\\tau_z - r_0/\\tau_0)`
    
        In addition to implementing these arithmetic operations, the
        method may also perform some bounding and damping.   
        
        :param z: Mean computed at the factor node
        :param zvar: Variance computed at the factor node
        :param r0: Mean of the incoming msg to the factor node
        :param rvar0:  Variance of the incoming msg to the factor node
        :param r1_prev: Previous mean of the msg from the factor node.
           This is used for damping messages.
        :param rvar1_prev:  Previous variance of the msg from the factor node.    
        
        :returns: :code:`r1,rvar1` mean and variance sent back to the 
            variable node.
        """
        raise NotImplementedError()
        
    def cost(self,d,dvar,rvar):
        """
        Computes the Gaussian cost in belief propagation.
        
        Nominally, for MAP estimation, this should compute:
        
        :math:`c = \|d\|^2/(2\\tau_r)
        
        For MMSE estimation, this computes,
        
        :math:`c = \|d\|^2/(2\\tau_r) + n\\tau_d/\\tau_r
        
        where :math:`d` is a estimation difference and :math:`\\tau_r`
        is a variance.
        
        :param d:  Estimation difference
        :param rvar:  Variance
        :returns: :code:`cost` the cost :math:`c` defined above.
        """
        raise NotImplementedError()
        
    def Hgauss(self, zvar):
        """
        Computes the Gaussian entropy for a given variance.
        
        For MAP estimation, this should return 0
        For MMSE estimation, this should return 
        
        :math:`H = (n/2)(1+\\log(2\\pi \\tau_z))`
        
        :param zvar:  Variance :math:`\\tau_z`
        :returns:  :code:`H` the Gaussian entropy
        """        
        raise NotImplementedError()
        
    
class VarHdlSimp(VarHdl):
    """
    Simple estimation variance handler.
    
    The handler assumes that the variance is a scalar.
    
    The method performs bounding by limiting :math:`\\alpha =\\tau_z/\\tau_0`,
    the ratio of the output to input variance.  Normally, 
    :math:`\\alpha \in [0,1]`.
    
    :param alpha_min:  Minimum value for :math:`\\alpha` above
    :param alpha_max:  Maximum value for :math:`\\alpha` above
    :param damp:  Damping constant.  :code:`damp=1` implies no
       damping.
    :param rep_axes:  The axes on which the variance is to be repeated.
       Default :code:`rep_axes=[]` implies the variance is not repeated.
    :param shape:  Shape of the estimation object on which the variance is 
       applied.
    :param Boolean is_compelx:  If data is complex
    :param Boolean is_compelx:  If the estimation is MAP or MMSE
    """
    def __init__(self, alpha_min=1e-5, alpha_max=1-1e-5, damp=1, rep_axes=[],\
                 shape = [], is_complex=False, map_est=True):
        VarHdl.__init__(self)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.damp = 1
        self.rep_axes = rep_axes
        self.is_complex = is_complex
        self.map_est = map_est
        self.shape = shape
            

    def msg_sub(self,z,zvar,r0,rvar0,r1_prev=[],rvar1_prev=[]):
        """
        Variance subtraction for message passing
        
        See base class :class:`VarHdl` for more details.
        
        :param z: Mean at the factor node
        :param zvar: Variance at the factor node
        :param r0: Mean of the incoming msg to the factor node
        :param rvar0:  Variance of the incoming msg to the factor node
        :param r1_prev: Previous mean of the msg from the factor node.
           This is used for damping messages.
        :param rvar1_prev:  Previous variance of the msg from the factor node.    
        """
        
        # Threshold the decrease in variance
        alpha = zvar/rvar0
        alpha = np.maximum(self.alpha_min, alpha)
        alpha = np.minimum(self.alpha_max, alpha)
        
        # Compute the message
        rvar1 = alpha/(1-alpha)*rvar0        
        r1 = (z-alpha*r0)/(1-alpha)
        
        # Apply the damping
        if (r1_prev != []) and (self.damp < 1):
            r1 = self.damp*r1 + (1-self.damp)*r1_prev
        if (rvar1_prev != []) and (self.damp < 1):
            rvar1 = self.damp*rvar1 + (1-self.damp)*rvar1_prev
                
        return r1, rvar1
        
    def cost(self,d,dvar,rvar):
        """
        Computes the Gaussian cost in belief propagation.
        
        See base class :class:`VarHdl` for more details.
        
        :param d:  Estimation difference
        :param rvar:  Variance
        :returns: :code:`cost` the cost :math:`c` defined above.
        """
        if self.rep_axes != []:
            rvar1 = common.repeat_axes(rvar,self.shape,self.rep_axes,rep=False) 
        else:
            rvar1 = rvar
        cost = np.sum((1/rvar1)*(np.abs(d)**2))
        
        if not self.map_est:
            cost += np.prod(d.shape)*np.mean(dvar/rvar)            
            
        if not self.is_complex:
            cost *= 0.5
        return cost
        
    def Hgauss(self, zvar):
        """
        Computes the Gaussian entropy for a given variance.
        
        See base class :class:`VarHdl` for more details.
        
        :param zvar:  Variance :math:`\\tau_z`
        :returns:  :code:`H` the Gaussian entropy
        """        
        if self.map_est:
            return 0
            
        H = np.prod(self.shape)*(1+np.mean(np.log(2*np.pi*zvar)))
        if not self.is_complex:
            H *= 0.5
        
        return H
        
        
        
        
        
