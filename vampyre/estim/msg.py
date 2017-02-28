"""
msg.py:  Basic classes for message passing
"""
from __future__ import division
from __future__ import print_function

import numpy as np

# Import other sub-packages
import vampyre.common as common

class MsgHdl(object):
    """
    Base class for Gaussian message passing handler
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
        
    def cost(self,z,zvar,r,rvar):
        """
        Computes the Gaussian cost in belief propagation.
        
        Nominally, for MAP estimation, this should compute the normalized
        difference:
        
        :math:`c = \|z-r\|^2/(2\\tau_r)
        
        For MMSE estimation, this computes,
        
        :math:`c = \|z-r\|^2/(2\\tau_r) + d\\tau_z/(2\\tau_r), 
        
        where :math:`d` is the dimension of the estimate.
                
        :param z:  Estimate :math:`z`
        :param r:  Estimate :math:`r`        
        :param zvar:  Variance :math:`\\tau_z`
        :param rvar:  Variance :math:`\\tau_r`
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
        
    
class MsgHdlSimp(MsgHdl):
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
    :param damp_lim:  Damping will not be performed if 
        :code:`rvar1_prev > damp_lim`
    :param rep_axes:  The axes on which the variance is to be repeated.
       Default :code:`rep_axes=[]` implies the variance is not repeated.
    :param shape:  Shape of the estimation object on which the variance is 
       applied.
    :param Boolean is_compelx:  If data is complex
    :param Boolean is_compelx:  If the estimation is MAP or MMSE
    :param var_scale:  Scales variance to improve robustness
    """
    def __init__(self, alpha_min=1e-5, alpha_max=1-1e-5, damp=0.95, rep_axes=[],\
                 shape = [], is_complex=False, map_est=True, damp_lim=1e6,\
                 var_scale=1):
        MsgHdl.__init__(self)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.damp = damp
        self.damp_lim = damp_lim
        self.rep_axes = rep_axes
        self.is_complex = is_complex
        self.map_est = map_est
        self.shape = shape
        self.var_scale = var_scale
            

    def msg_sub(self,z,zvar,r0,rvar0,r1_prev=None,rvar1_prev=None):
        """
        Variance subtraction for message passing
        
        See base class :class:`MsgHdl` for more details.
        
        :param z: Mean at the factor node
        :param zvar: Variance at the factor node
        :param r0: Mean of the incoming msg to the factor node
        :param rvar0:  Variance of the incoming msg to the factor node
        :param r1_prev: Previous mean of the msg from the factor node.
           This is used for damping messages.
        :param rvar1_prev:  Previous variance of the msg from the factor node.    
        """
        
        # Check the infinite variance case
        if np.any(rvar0 == np.Inf):
            r1 = z
            rvar1 = zvar
            return r1, rvar1
            
        # Return infinite variance (i.e. non-informative data) if the variance
        # did not sufficiently decrease
        alpha = zvar/rvar0
        if np.any(alpha >= self.alpha_max):
            if np.isscalar(rvar0):
                rvar1 = np.Inf
            else:
                rvar1 = np.Inf*np.ones(rvar0.shape)
            r1 = z
            return r1, rvar1
        
        # Threshold the decrease in variance
        alpha = zvar/rvar0
        alpha = np.maximum(self.alpha_min, alpha)
        alpha = np.minimum(self.alpha_max, alpha)
        
        # Compute the message
        rvar1 = alpha/(1-alpha)*rvar0        
        r1 = (z-alpha*r0)/(1-alpha)
        
        # Apply the damping
        if not (r1_prev is None) and not (rvar1_prev is None) and (self.damp < 1):
            if np.all(rvar1_prev < self.damp_lim):
                r1 = self.damp*r1 + (1-self.damp)*r1_prev
                rvar1 = self.damp*rvar1 + (1-self.damp)*rvar1_prev
                
        # Boost the variance
        rvar1 = rvar1*self.var_scale
                                
        return r1, rvar1
        
    def cost(self,z,zvar,r,rvar):
        """
        Computes the Gaussian cost in belief propagation.
        
        See base class :class:`MsgHdl` for more details.
        
        :param z:  Estimate :math:`z`
        :param r:  Estimate :math:`r`        
        :param zvar:  Variance :math:`\\tau_z`
        :param rvar:  Variance :math:`\\tau_r`
        :returns: :code:`cost` the cost :math:`c` defined above.
        """
        if self.rep_axes != []:
            rvar1 = common.repeat_axes(rvar,self.shape,self.rep_axes,rep=False) 
        else:
            rvar1 = rvar
        cost = np.sum((1/rvar1)*(np.abs(z-r)**2))
        
        if not self.map_est:
            cost += np.prod(self.shape)*np.mean(zvar/rvar)            
            
        if not self.is_complex:
            cost *= 0.5
        return cost
        
    def Hgauss(self, zvar):
        """
        Computes the Gaussian entropy for a given variance.
        
        See base class :class:`MsgHdl` for more details.
        
        :param zvar:  Variance :math:`\\tau_z`
        :returns:  :code:`H` the Gaussian entropy
        """        
        if self.map_est:
            return 0
            
        H = np.prod(self.shape)*(1+np.mean(np.log(2*np.pi*zvar)))
        if not self.is_complex:
            H *= 0.5
        
        return H
        
        
class ListMsgHdl(MsgHdl):
    """
    Message handler for a list of messages
    
    :param hdl_list:  List of variance handlers
    """
    pass
    
    def __init__(self, hdl_list):
        self.hdl_list = hdl_list
            

    def msg_sub(self,z,zvar,r0,rvar0,r1_prev=[],rvar1_prev=[]):
        """
        Variance subtraction for message passing
        
        See base class :class:`MsgHdl` for more details.
        
        :param z: Mean at the factor node
        :param zvar: Variance at the factor node
        :param r0: Mean of the incoming msg to the factor node
        :param rvar0:  Variance of the incoming msg to the factor node
        :param r1_prev: Previous mean of the msg from the factor node.
           This is used for damping messages.
        :param rvar1_prev:  Previous variance of the msg from the factor node.    
        """

        # Create an empty list 
        r1 = []
        rvar1 = []
        
        # Loop over the message handlers
        i = 0        
        for hdl in self.hdl_list:
            
            # Get the items from the i-th element
            zi = z[i]
            zvari = zvar[i]
            r0i = r0[i]
            rvar0i = rvar0[i]
            if r1_prev == []:
                r1_previ = []
            else:
                r1_previ = r1_prev[i]            
            if rvar1_prev == []:
                rvar1_previ = []
            else:                    
                rvar1_previ = rvar1_prev[i]
                
            # Call the message handler
            r1i, rvar1i = hdl.msg_sub(zi,zvari,r0i,rvar0i,r1_previ,rvar1_previ)
            
            # Add to the output list
            r1.append(r1i)
            rvar1.append(rvar1i)        
            i += 1
                
        return r1, rvar1
        
    def cost(self,z,zvar,r,rvar):
        """
        Computes the Gaussian cost in belief propagation.
        
        See base class :class:`MsgHdl` for more details.
        
        :param z:  Estimate :math:`z`
        :param r:  Estimate :math:`r`        
        :param zvar:  Variance :math:`\\tau_z`
        :param rvar:  Variance :math:`\\tau_r`
        :returns: :code:`cost` the cost :math:`c` defined above.
        """        
        
        # Loop over the message handlers and accumulate the cost
        cost = 0
        i = 0        
        for hdl in self.hdl_list:
            
            # Get the items from the i-th element
            zi = z[i]
            ri = r[i]
            rvari = rvar[i]
            zvari = zvar[i]            
            
            # Call the message handler
            ci = hdl.cost(zi,zvari,ri,rvari)
            
            # Add to the cost
            cost += ci
            i += 1
                
        return cost
        
    def Hgauss(self, zvar):
        """
        Computes the Gaussian entropy for a given variance.
        
        See base class :class:`VarHdl` for more details.
        
        :param zvar:  Variance :math:`\\tau_z`
        :returns:  :code:`H` the Gaussian entropy
        """        
        # Loop over the message handlers and accumulate the Gaussian entropy
        H = 0
        i = 0        
        for hdl in self.hdl_list:
            zvari = zvar[i]            
            H += hdl.Hgauss(zvari)
            i += 1
        return H        
        
