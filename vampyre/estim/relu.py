"""
relu.py:  Estimator and test for the rectified linear unit
"""
from __future__ import division
from __future__ import print_function

import numpy as np

# Import other subpackages in vampyre
import vampyre.common as common

# Import individual classes and methods from the current subpackage
from vampyre.estim.base import Estim
from vampyre.estim.interval import gauss_integral
       
class ReLUEstim(Estim):    
    """
    Estimatar for a rectified linear unit
    
    :math:`z_1 = f(z_0) = \\max(0,z_0)`
    
    :param shape:  shape of :math:`z_0` and :math:`z_1`
    :param z0rep_axes:  axes on which the variance for :math:`z_0` is averaged
    :param z1rep_axes:  axes on which the variance for :math:`z_1` is averaged
    :param map_est: Flag indicating if estimation is MAP or MMSE.
    """        
    def __init__(self,shape,z0rep_axes=(0,), z1rep_axes=(0,), map_est=False):
        Estim.__init__(self)
        self.shape = shape
        ndim = len(shape)
        if z0rep_axes == 'all':
            z0rep_axes = tuple(range(ndim))
        if z1rep_axes == 'all':
            z1rep_axes = tuple(range(ndim))            
        self.z0rep_axes = z0rep_axes
        self.z1rep_axes = z1rep_axes
        self.cost_avail = True
        self.map_est = map_est
        
        # Initial variances
        self.zvar0_init= np.Inf
        self.zvar1_init= np.Inf
    
    def est_init(self,return_cost=False):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.
                
        :param Boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """          
        zhat0   = np.zeros(self.shape)
        zhat1   = np.zeros(self.shape)
        zhat    = [zhat0,zhat1]
     
        # Compute the shapes for the variance and set the initial value of
        # the variance according to the shape
        ndim = len(self.shape)
        axes_spec = [i for i in range(ndim) if i not in self.z0rep_axes]
        if axes_spec == []:
            zvar0 = self.zvar0_init
        else:
            shape1 = tuple(np.array(self.shape)[axes_spec])
            zvar0 = np.tile(self.zvar0_init, shape1)
        axes_spec = [i for i in range(ndim) if i not in self.z1rep_axes]
        if axes_spec == []:
            zvar1 = self.zvar1_init
        else:
            shape1 = tuple(np.array(self.shape)[axes_spec])
            zvar1 = np.tile(self.zvar1_init, shape1)            
        zvar = [zvar0,zvar1]

        cost = 0
        if return_cost:
            return zhat, zvar, cost
        else:
            return zhat, zvar            
            
    def est(self,r,rvar,return_cost=False):
        """
        Estimation function
        
        The proximal estimation function as 
        described in the base class :class:`vampyre.estim.base.Estim`
                
        :param r: Proximal mean
        :param rvar: Proximal variance
        :param boolean return_cost:  Flag indicating if :code:`cost` 
            is to be returned
         
        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior 
        mean, variance and optional cost.
        """        
        if self.map_est:
            return self.est_map(r,rvar,return_cost)
        else:
            return self.est_mmse(r,rvar,return_cost)
                        
    
    def est_map(self,r,rvar,return_cost):
        """
        MAP Estimation
        In this case,  we wish to minimize
            cost = (z0-r0)^2/(2*rvar0) + (z1-r1)^2/(2*rvar1)
        
        where z1 = max(0,z0) 
        """
        # Unpack the terms
        r0, r1 = r
        rvar0, rvar1 = rvar
        
        # Clip variances
        rvar1 = np.minimum(1e8*rvar0, rvar1)
        
        # Reshape the variances
        rvar0 = common.repeat_axes(rvar0,self.shape,self.z0rep_axes)
        rvar1 = common.repeat_axes(rvar1,self.shape,self.z1rep_axes)
        
        # Positive case:  z0 >= 0 and hence z1=z0
        z0p = np.maximum(0, (rvar0*r1 + rvar1*r0)/(rvar0 + rvar1))
        z1p = z0p
        zvar0p = rvar0*rvar1/(rvar0+rvar1)
        zvar1p = zvar0p
        costp = 0.5*((z0p-r0)**2/rvar0 + (z1p-r1)**2/rvar1)
        
        # Negative case:  z0 <= 0 and hence z1 = 0
        z0n = np.minimum(0, r0)
        z1n = 0
        zvar0n = rvar0
        zvar1n = 0
        costn = 0.5*((z0n-r0)**2/rvar0 + (z1n-r1)**2/rvar1)
        
        # Find lower cost and select the correct choice for each element        
        Ip = (costp < costn)
        zhat0 = z0p*Ip + z0n*(1-Ip)
        zhat1 = z1p*Ip + z1n*(1-Ip)
        zhatvar0 = zvar0p*Ip + zvar0n*(1-Ip)
        zhatvar1 = zvar1p*Ip + zvar1n*(1-Ip)
        cost = np.sum(costp*Ip + costn*(1-Ip))
                        
        # Average the variance over the specified axes
        zhatvar0 = np.mean(zhatvar0,axis=self.z0rep_axes)
        zhatvar1 = np.mean(zhatvar1,axis=self.z1rep_axes)
        zhatvar = [zhatvar0,zhatvar1]
        
        # Pack the first order terms
        zhat = [zhat0,zhat1]

        if not return_cost:        
            return zhat, zhatvar
        else:
            return zhat, zhatvar, cost
        
    def est_mmse(self,r,rvar,return_cost):                
        """        
        In the MMSE estimation case, we wish to estimate
        z0 and z1 with priors zi = N(ri,rvari) and z1=f(z0)
        
        Substituting in z1 = f(z0), we have the density of z0:
          
           p(z0)  \propto qn(z0)1_{z0 < 0}  + qp(z0)1_{z0 > 0}
           
        where
           
           qp(z0)  = exp[-(z0-r0)^2/(2*rvar0) - (z0-r1)^2/(2*rvar1)]
           qn(z0)  = exp[-(z0-r0)^2/(2*rvar0) - r1^2/(2*rvar1)]
           
        First, we complete the squares and write:
        
           qp(z0) = exp(Amax)*Cp*exp(-(z0-rp)^2/(2*zvarp))/sqrt(2*pi)  
           qn(z0) = exp(Amax)*Cn*exp(-(z0-rn)^2/(2*zvarn))/sqrt(2*pi)        
           
        """


        # Unpack the terms
        r0, r1 = r
        rvar0, rvar1 = rvar
                
        # Reshape the variances
        rvar0 = common.repeat_axes(rvar0,self.shape,self.z0rep_axes)
        rvar1 = common.repeat_axes(rvar1,self.shape,self.z1rep_axes)
        
        if np.any(rvar1 == np.Inf):
            # Infinite variance case.
            zvarp = rvar0
            zvarn = rvar0
            rp = r0
            rn = r0
            Cp = 1
            Cn = 1
            Amax = 0
                        
        else:
            
            # Compute the MAP estimate
            zhat_map, zvar_map = self.est_map(r,rvar,return_cost=False)
            zhat0_map, zhat1_map = zhat_map
            zvar0_map, zvar1_map = zvar_map
                            
            # Compute the conditional Gaussian terms for z > 0 and z < 0
            zvarp = rvar0*rvar1/(rvar0+rvar1)
            zvarn = rvar0
            rp = (rvar1*r0 + rvar0*r1)/(rvar0+rvar1)        
            rn = r0
    
            # Compute scaling constants for each region
            Ap = 0.5*((rp**2)/zvarp - (r0**2)/rvar0 - (r1**2)/rvar1)
            An = 0.5*(-(r1**2)/rvar1)
            Amax = np.maximum(Ap,An)
            Ap = Ap - Amax
            An = An - Amax
            Cp = np.exp(Ap)
            Cn = np.exp(An)
                            
        # Compute moments for each region
        zp = Cp*gauss_integral(0, np.Inf, rp, zvarp)
        zn = Cn*gauss_integral(-np.Inf, 0, rn, zvarn)
        
        # Find poorly conditioned points        
        Ibad = (zp[0] + zn[0] < 1e-6)
        zpsum = zp[0] + zn[0] + Ibad
        
        # Compute mean        
        zhat0 = (zp[1] + zn[1])/zpsum
        zhat1 = zp[1]/zpsum
        
        # Compute the variance
        zhatvar0 = (zp[2] + zn[2])/zpsum - zhat0**2
        zhatvar1 = zp[2]/zpsum - zhat1**2
        
        # Replace bad points with MAP estimate
        if 1:
            zhat0 = zhat0*(1-Ibad) + zhat0_map*Ibad
            zhat1 = zhat1*(1-Ibad) + zhat1_map*Ibad
            zhatvar0 = zhatvar0*(1-Ibad) + zvar0_map*Ibad
            zhatvar1 = zhatvar1*(1-Ibad) + zvar1_map*Ibad
        zhat = [zhat0,zhat1]     
        
        # Average the variance over the specified axes
        zhatvar0 = np.mean(zhatvar0,axis=self.z0rep_axes)
        zhatvar1 = np.mean(zhatvar1,axis=self.z1rep_axes)
        zhatvar = [zhatvar0,zhatvar1]

        if not return_cost:        
            return zhat, zhatvar
            
        """
        Compute the 
            cost = -\log \int p(z_0) 
                 = -Amax - log(zp[0] + zn[0])        
        """
        nz = np.prod(self.z0rep_axes)
        cost = -nz*np.mean(Amax - np.log(zpsum))
        return zhat, zhatvar, cost
        
    