"""
gaussian.py:  Classes for Gaussian estimation
"""
from __future__ import division
from __future__ import print_function

import numpy as np
from vampyre.common.utils import repeat_axes, repeat_sum
from vampyre.common.utils import TestException
from vampyre.estim.base import Estim

class GaussEst(Estim):
    """ Gaussian estimator class
    
    Estimator for a Gaussian penalty 
    :math:`f(z)=(1/2 \\tau_z)\|z-\\bar{z}\|^2+(1/2)\\ln(2\pi \\tau_z)`
    When :math:`z` is complex, the factor :math:`1/2` is removed.
        
    :param zmean:  prior mean, :math:`\\bar{z}`
    :param zvar:   prior variance, :math:`\\tau_z`  
    :param shape:  shape of :math:`z`
    :param var_axes:  axes on which the prior variance is repeated.
         This is also the axes on which the variance is averaged.
         (default=`all` indicating all axes are averaged, meaning that
         the variance is a scalar)
    :param zmean_axes:  axis on which the prior mean is
         repeated.  (default=`all` indicating prior mean is repeated
         on all axes, meaning that the prior mean is a scalar)
    :param Boolean is_complex:  indiates if :math:`z` is complex    
    :param Boolean map_est:  indicates if estimator is to perform MAP 
        or MMSE estimation. This is used for the cost computation.
    """    
    def __init__(self, zmean, zvar, shape, 
                 var_axes = (0,), zmean_axes='all',
                 is_complex=False, map_est=False):
        Estim.__init__(self)
        self.zmean = zmean
        self.zvar = zvar
        self.cost_avail = True  
        self.is_complex = is_complex  
        self.map_est = map_est         
        self.shape = shape if (type(shape) is not int) else (shape,)
        self.var_axes = var_axes
        self.zmean_axes = zmean_axes
        
        ndim = len(self.shape)
        if self.var_axes == 'all':
            self.var_axes = tuple(range(ndim))        
        if self.zmean_axes == 'all':
            self.zmean_axes = tuple(range(ndim))
            
        # If zvar is a scalar, then repeat it to the required shape,
        # which are all the dimensions not being averaged over
        if np.isscalar(self.zvar):
            ndim = len(self.shape)
            axes_spec = [i for i in range(ndim) if i not in self.var_axes]
            if axes_spec != []:
                shape1 = tuple(np.array(self.shape)[axes_spec])
                self.zvar = np.tile(self.zvar, shape1)
        
                 
        
    def est_init(self, return_cost=False, avg_var_cost=True):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.
        
        :param boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :param Boolean avg_var_cost: Average variance and cost.
            This should be disabled to obtain per element values.
            (Default=True)
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """        
        zmean = repeat_axes(self.zmean, self.shape, self.zmean_axes)
        zvar  = self.zvar
        if not avg_var_cost:
            zvar = repeat_axes(zvar, self.shape, self.var_axes)
        if not return_cost:
            return zmean, zvar
            
        # Cost including the normalization factor
        if self.map_est:
            clog = np.log(2*np.pi*self.zvar)
            if avg_var_cost:
                cost = repeat_sum(clog, self.shape, self.var_axes)
        else:
            cost = 0
        if not self.is_complex:
            cost = 0.5*cost
        return zmean, zvar, cost
                    
    def est(self,r,rvar,return_cost=False,avg_var_cost=True):
        """
        Estimation function
        
        The proximal estimation function as 
        described in the base class :class:`vampyre.estim.base.Estim`
                
        :param r: Proximal mean
        :param rvar: Proximal variance
        :param Boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :param Boolean avg_var_cost: Average variance and cost.
            This should be disabled to obtain per element values.
            (Default=True)
        
        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior
            mean, variance and optional cost.
        """
        
        # Infinite variance case
        if np.any(rvar==np.Inf):
            return self.est_init(return_cost, avg_var_cost)
                    
        zhatvar = rvar*self.zvar/(rvar + self.zvar)
        gain = self.zvar/(rvar + self.zvar)
        gain = repeat_axes(gain,self.shape,self.var_axes,rep=False) 
        if not avg_var_cost:
            zhatvar = repeat_axes(zhatvar,self.shape,self.var_axes) 
        
        zhat = gain*(r-self.zmean) + self.zmean
        
        if not return_cost:                
            return zhat, zhatvar
            
        # Computes the MAP cost
        zvar1 = repeat_axes(self.zvar,self.shape,self.var_axes,rep=False)
        rvar1 = repeat_axes(rvar,     self.shape,self.var_axes,rep=False)
        cost = (np.abs(zhat-self.zmean)**2) / zvar1 \
             + (np.abs(zhat-r)**2) / rvar1
        if avg_var_cost:
            cost = np.sum(cost)
            
        # Compute cost
        nz = np.prod(self.shape)
        if self.map_est:
            clog =  np.log(2*np.pi*self.zvar) 
            if avg_var_cost:
                clog = np.mean(clog)*nz
            else:
                clog = np.log(2*np.pi*zvar1)
            cost += clog
        else:
            d = np.log(self.zvar/zhatvar) 
            if avg_var_cost:
                cost += np.mean(d)*nz
            else:
                cost += d                
            
        # Scale for real case
        if not self.is_complex:
            cost = 0.5*cost            
        return zhat, zhatvar, cost
            