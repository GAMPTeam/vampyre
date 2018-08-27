"""
relu.py:  Estimator and test for the rectified linear unit
"""
from __future__ import division
from __future__ import print_function

import numpy as np

# Import other subpackages in vampyre
import vampyre.common as common

# Import individual classes and methods from the current subpackage
from vampyre.estim.base import BaseEst
from vampyre.estim.interval import gauss_integral
       
class ReLUEst(BaseEst):    
    """
    Estimatar for a rectified linear unit
    
    :math:`z_1 = f(z_0) = \\max(0,z_0)`
    
    :param shape:  shape of :math:`z_0` and :math:`z_1`
    :param var_axes:  List :code:`[var_axes[0],var_axes[1]]` of the 
         axes on which the input and output variances are averaged
    :param map_est: Flag indicating if estimation is MAP or MMSE.
    :param name:  Estimator name.
    """        
    def __init__(self,shape,var_axes=[(0,),(0,)], name=None, map_est=False):
        self.map_est = map_est
        
        # Initial variances
        self.zvar0_init= np.Inf
        self.zvar1_init= np.Inf
        
        nvars = 2
        dtype = np.float64
        BaseEst.__init__(self,shape=[shape,shape], var_axes=var_axes, dtype=dtype, name=name,\
            type_name='ReLUEst', nvars=nvars, cost_avail=True)        

    
    def est_init(self,return_cost=False,ind_out=None, avg_var_cost=True):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.
                
        :param Boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """     
        # Check parameters
        if ind_out is None:
            ind_out = [0,1]
        if not avg_var_cost:
            raise ValueError("disabling variance averaging not supported for ReLUEST") 
        zmean = []
        zvar = []
        if 0 in ind_out:
            zmean0 = np.zeros(self.shape[0])
            zvar0_shape = common.utils.get_var_shape(self.shape[0], self.var_axes[0])
            zvar0 = np.tile(self.zvar0_init, zvar0_shape)
            zmean.append(zmean0)
            zvar.append(zvar0)
        if 1 in ind_out:
            zmean1 = np.zeros(self.shape[1])
            zvar1_shape = common.utils.get_var_shape(self.shape[1], self.var_axes[1])
            zvar1 = np.tile(self.zvar1_init, zvar1_shape)
            zmean.append(zmean1)
            zvar.append(zvar1)     
        cost = 0
        if return_cost:
            return zmean, zvar, cost
        else:
            return zmean, zvar            
            
    def est(self,r,rvar,return_cost=False,ind_out=None, avg_var_cost=True):
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
        # Check parameters
        if ind_out is None:
            ind_out = [0,1]
        if not avg_var_cost:
            raise ValueError("disabling variance averaging not supported for ReLUEST") 

        if self.map_est:
            return self.est_map(r,rvar,return_cost,ind_out)
        else:
            return self.est_mmse(r,rvar,return_cost,ind_out)
                        
    
    def est_map(self,r,rvar,return_cost,ind_out):
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
        rvar0 = common.repeat_axes(rvar0,self.shape[0],self.var_axes[0])
        rvar1 = common.repeat_axes(rvar1,self.shape[1],self.var_axes[1])
        
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
        zhatvar0 = np.mean(zhatvar0,axis=self.var_axes[0])
        zhatvar1 = np.mean(zhatvar1,axis=self.var_axes[1])
        zhatvar = [zhatvar0,zhatvar1]
        
        # Pack the items
        zhat = []
        zhatvar = []
        if 0 in ind_out:
            zhat.append(zhat0)
            zhatvar.append(zhatvar0)
        if 1 in ind_out:
            zhat.append(zhat1)
            zhatvar.append(zhatvar1)

        if not return_cost:        
            return zhat, zhatvar
        else:
            return zhat, zhatvar, cost
        
    def est_mmse(self,r,rvar,return_cost,ind_out):                
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
        rvar0 = common.repeat_axes(rvar0,self.shape[0],self.var_axes[0])
        rvar1 = common.repeat_axes(rvar1,self.shape[1],self.var_axes[1])
        
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
            zhat_map, zvar_map = self.est_map(r,rvar,return_cost=False,ind_out=[0,1])
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
        
        # Average the variance over the specified axes
        zhatvar0 = np.mean(zhatvar0,axis=self.var_axes[0])
        zhatvar1 = np.mean(zhatvar1,axis=self.var_axes[1])
        
        # Pack the items
        zhat = []
        zhatvar = []
        if 0 in ind_out:
            zhat.append(zhat0)
            zhatvar.append(zhatvar0)
        if 1 in ind_out:
            zhat.append(zhat1)
            zhatvar.append(zhatvar1)


        if not return_cost:        
            return zhat, zhatvar
            
        """
        Compute the 
            cost = -\log \int p(z_0) 
                 = -Amax - log(zp[0] + zn[0])        
        """
        nz = np.prod(self.shape[0])
        cost = -nz*np.mean(Amax - np.log(zpsum))
        return zhat, zhatvar, cost
        
    