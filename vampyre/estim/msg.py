"""
msg.py:  Basic classes for message passing
"""
from __future__ import division
from __future__ import print_function

import numpy as np

# Import other sub-packages
import vampyre.common as common

class MsgHdl(object):
    def __init__(self):
        """
        Base class for Gaussian message passing handler
        """
        pass
    
    def init_msg(self,r,rvar,idir):
        """
        Overwrites an initial message with pre-set values.
        The values :code:`r,rvar` are the initial mean and variance
        to be sent to the factor node idir.  The default implementation simply
        returns these values.  
        
        :param r: Default initial mean to the factor node
        :param rvar: Default initial variance to the factor node 
        :param idir:  Index of the factor node.  
        :returns: :code:`r,rvar` Overwritten mean and variance.             
        """
        return r, rvar
    
    def msg_sub(self,z,zvar,idir):
        """
        Variance subtraction for message passing
        
        Suppose that a factor node computes a 
        mean and variance :math:`(z,\\tau_z)` based on an 
        incoming Gaussian message with parameters :math:`(r_0,\\tau_0)`.  
        This method then computes the outgoing message parameters 
        :math:`(r_1,\\tau_1)` to the other factor node using the subtraction
        
        :math:`\\tau_1 = \\tau_0\\tau_z / (\\tau_0 - \\tau_z)`
        
        :math:`r_1 = \\tau_1( z/\\tau_z - r_0/\\tau_0)`
    
        In addition to implementing these arithmetic operations, the
        method may also perform some bounding and damping.   The values
        for the outgoing messages :math:`(r_0,\\tau_0)` must be stored.
        
        :param z: Mean computed at the factor node
        :param zvar: Variance computed at the factor node
        :param idir: Index of the direction being used.
        
        :returns: :code:`r1,rvar1` mean and variance sent back to the 
            variable node.
        """
        raise NotImplementedError()
        
        
    def get_cost_terms(self):
        """
        Computes the node cost, gradient and constraint.
        
        This is an optional method that returns cost terms for tracking the 
        progress of the optimization.  
        
        Nominally, for MAP estimation, the node cost in the direction 
        :math:`i` is given by
        
        :math:`c[i] = \|z[i]-r[i]\|^2/(2\\tau_r[i])`,
        
        
        For MMSE estimation, this computes,
        
        :math:`c[i] = \|z[i]-r[i]\|^2/(2\\tau_r[i]) + d\\tau_z[i]/(2\\tau_r[i])`, 
        
        where :math:`d` is the dimension of the estimate.
        For both MAP and MMSE estimation, the gradient is given by
        
        :math:`g[i] = (z[i]-r[i])/(2\\tau_r[i])`
        
        and the constraint error is given by 
        
        :math:`s = z[0]-z[1]`.
                
        :returns: :code:`cost,grad,con` where :code:`cost` is the list
           of the two costs :math:`c[0], c[1]`.  :code:`grad` is the
           mean squared norm of :math:`g` and :code:`con` is the mean
           squared norm of :math:`s` above.
        
        """        
        cost = [0,0]
        grad = 0
        con = 0
        return cost, grad, con
        
        
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
    :param damp:  Damping constant for the first-order terms.  
       :code:`damp=1` implies no damping.
    :param damp_var:  Damping constant for the second-order terms.  
       :code:`damp_var=1` implies no damping.   :code:`damp_var=0` will
       not update the variance and keep the values at rvar_init.
    :param rep_axes:  The axes on which the variance is to be repeated.
       Default :code:`rep_axes=[]` implies the variance is not repeated.
    :param shape:  Shape of the estimation object on which the variance is 
       applied.
    :param Boolean is_complex:  If data is complex
    :param Boolean map_est:  If the estimation is MAP or MMSE
    :param var_scale:  Scales variance to improve robustness
    :param rvar_min:  Minimum variance
    :param rvar_max:  Maximum variance
    :param rinit:  Initial mean value for each direction which is a given as a
       list of two items, one for each direction.    A value of 
       :code:`[None,None]` indicates that no value is given.
    :param rvar_init:  Initial variance for each direction.    
    """
    def __init__(self, alpha_min=1e-5, alpha_max=1-1e-5, damp=0.95, damp_var=0.95,rep_axes=(0,),\
                 shape = [], is_complex=False, map_est=True,\
                 var_scale=1,rvar_min=0,rvar_max=1e5, rinit=[None,None],rvar_init=[None,None]):
        MsgHdl.__init__(self)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.damp = damp
        self.damp_var = damp_var            
        self.rep_axes = rep_axes
        self.is_complex = is_complex
        self.map_est = map_est
        self.shape = shape
        self.var_scale = var_scale
        self.rvar_min = rvar_min
        self.rvar_max = rvar_max

        # Number of directions on which messages are passed, which is equal to 
        # the number of factor nodes attached to the message node.  Right now,
        # this handler only supports two directions
        self.ndir = 2
        
        # Convert scalar values to appropriate shape
        self.rinit = [None,None]
        self.rvar_init = [None,None]
        self.rprev = [None,None]
        self.rvar_prev = [None,None]
        for i in range(self.ndir):
            if not (rinit[i] is None):
                self.rinit[i] = np.copy(rinit[i])
            if not (rvar_init[i] is None):
                self.rvar_init[i] = np.copy(rvar_init[i])                
            if np.isscalar(self.rinit[i]):
                self.rinit[i] = common.repeat_const(\
                    self.rinit[i],self.shape,self.rep_axes)
            if np.isscalar(self.rvar_init[i]):
                self.rvar_init[i] = common.repeat_const(\
                    self.rvar_init[i],self.shape,self.rep_axes)
            if not (self.rinit[i] is None):
                self.rprev[i] = np.copy(self.rinit[i])
            if not (self.rvar_init[i] is None):
                self.rvar_prev[i] = np.copy(self.rvar_init[i])
            
        
        # Initialize the past variances and means for the purpose of damping
        self.zprev = [None,None]
        self.zvar_prev = [None,None]
                        
        # Initialize the cost terms
        self.cost = [0,0]
        self.grad = [np.zeros(self.shape), np.zeros(self.shape)]
        self.con = 0
        
        
    def init_msg(self,r,rvar,idir):
        """
        Overwrites an initial message with pre-set values.
        The values :code:`r,rvar` are the initial mean and variance
        to be sent to the factor node idir.  The default implementation simply
        returns these values.  In this implementation, they are overwritten
        if the values :code:`rinit,rvar_init` if they are not set to :code:`None`.
        
        :param r: Default initial mean to the factor node
        :param rvar: Default initial variance to the factor node 
        :param idir:  Index of the factor node.  
        :returns: :code:`e,rvar` Overwritten mean and variance.             
        """
        if self.rinit[idir] is None:
            r1 = r            
        else:
            r1 = np.copy(self.rinit[idir])
        if self.rvar_init[idir] is None:
            rvar1 = rvar
        else:
            rvar1 = np.copy(self.rvar_init[idir])

        # Bound the variance
        rvar1 = np.maximum(rvar1, self.rvar_min)
        rvar1 = np.minimum(rvar1, self.rvar_max)
            
        self.rprev[idir] = np.copy(r1)
        self.rvar_prev[idir] = np.copy(rvar1)        
        return r1, rvar1
        
                
    def msg_sub(self,z,zvar,idir=0):
        """
        Variance subtraction for message passing
        
        See base class :class:`MsgHdl` for more details.
        
        :param z: Mean from the factor node
        :param zvar: Variance from the factor node
        :param idir: Index of the factor node for incoming message
        """

        # If variance is fixed, then overwrite the variance to be consistent 
        # with the prior variances.            
        if self.damp_var == 0:            
            zvar = self.rvar_prev[0]*self.rvar_prev[1]/(self.rvar_prev[0] + self.rvar_prev[1])
        
        # Save the z value
        self.zprev[idir] = z
        self.zvar_prev[idir] = zvar
        
        # Get variance passed to the incoming node
        r0 = self.rprev[idir]
        rvar0 = self.rvar_prev[idir]
        
        
        if rvar0 is None:
            # Special case where there is no previous variance
            rvar1 = zvar
            r1 = z
        else:
            
            # Compute cost and gradient
            self.compute_cost_terms(idir)
                        
            # Threshold the decrease in variance
            alpha = zvar/rvar0
            alpha = np.maximum(self.alpha_min, alpha)
            alpha = np.minimum(self.alpha_max, alpha)
                
            # Compute output variance
            rvar1 = alpha/(1-alpha)*rvar0
                                                                                        
            # Compute the message
            alpha_rep = common.repeat_axes(alpha,self.shape,self.rep_axes,rep=False)
            r1 = (z-alpha_rep*r0)/(1-alpha_rep)
        
        # Bound the variance
        rvar1 = np.maximum(rvar1, self.rvar_min)
        rvar1 = np.minimum(rvar1, self.rvar_max)
            
        # Get the last output mean and variance
        jdir = (idir+1)%2
        rvar1_prev = self.rvar_prev[jdir]
        r1_prev = self.rprev[jdir]
        if not (r1_prev is None):
            r1 = self.damp*r1 + (1-self.damp)*r1_prev
        if not (rvar1_prev is None):
            gam1_prev = 1/rvar1_prev
            gam1 = self.damp_var/rvar1 + (1-self.damp_var)*gam1_prev
            rvar1 = 1/gam1
            
        # Bound the variance
        rvar1 = np.maximum(rvar1, self.rvar_min)
        rvar1 = np.minimum(rvar1, self.rvar_max)
                        
        # Save outgoing message
        self.rprev[jdir] = r1
        self.rvar_prev[jdir] = rvar1
                                
        return r1, rvar1
        
    def compute_cost_terms(self,idir):
        """
        Computes the Gaussian cost in belief propagation.
        
        See base class :class:`MsgHdl` for more details.
        
        :param z:  Estimate :math:`z`
        :param r:  Estimate :math:`r`        
        :param zvar:  Variance :math:`\\tau_z`
        :param rvar:  Variance :math:`\\tau_r`
        :returns: :code:`cost` the cost :math:`c` defined above.
        """
        
        # Skip update if 
        if self.rvar_prev[idir] is None:
            return
            
        rvar_rep = common.repeat_axes(self.rvar_prev[idir],\
            self.shape,self.rep_axes,rep=False) 
            
        # Computes the gradient
        z = self.zprev[idir]
        r = self.rprev[idir]        
        self.grad[idir] = (z-r)/rvar_rep
        self.cost[idir] = np.sum((1/rvar_rep)*(np.abs(z-r)**2))
        
        if not self.map_est:
            self.cost[idir] += np.prod(self.shape)*\
                np.mean(self.zvar_prev[idir]/self.rvar_prev[idir])            
            
        if not self.is_complex:
            self.cost[idir] *= 0.5
        
    def get_cost_terms(self):
        """
        Retrieves the computed node cost, gradient and constraint.
        See base class :class:`MsgHdl` for more details.
        
        :returns: :code:`cost,grad,con` where :code:`cost` is the list
           of the two costs :math:`c[0], c[1]`.  :code:`grad` is the
           mean squared norm of :math:`g` and :code:`con` is the mean
           squared norm of :math:`s` above.
        
        """
        
        # Average gradient
        grad = np.mean( np.abs(self.grad[0]+self.grad[1])**2 )
        
        # Average constraint
        con = np.mean( np.abs( self.zprev[0]-self.zprev[1])**2 )
        
        return self.cost, grad, con
        
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
    Message handler for a list of messages.
    
    This needs to be fixed for the new message structure.
    
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
        
