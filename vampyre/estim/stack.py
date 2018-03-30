"""
stack.py:  Concantenated or "stacked" estimators
"""
from __future__ import division
from __future__ import print_function
import numpy as np

# Import individual classes and methods from the current subpackage
from vampyre.estim.base import BaseEst

class StackEst(BaseEst):
    """
    A stacked estimator
    
    This is an estimator for a list of variables 
    :math:`z=[z_0,\\ldots,z_{K-1}]`.  
    
    :param est_list:  List of estimators for each variable
    """
    def __init__(self,est_list,name=None):

        # Save estimator list
        self.est_list = est_list
        
        # Get parameters from the estimators
        shape = []
        var_axes = []
        dtype = []        
        cost_avail = True
        nvars = len(est_list)
        for est in est_list:
            shape.append(est.shape)
            var_axes.append(est.var_axes)
            dtype.append(est.dtype)
            cost_avail = cost_avail and est.cost_avail
        self.cost_avail = cost_avail
        BaseEst.__init__(self,shape=shape,var_axes=shape,dtype=dtype,\
            name=name, type_name='StackEst', nvars=nvars, cost_avail=cost_avail)

        
    def est_init(self, return_cost=False,ind_out=None,\
        avg_var_cost=True):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.
              
        :param boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """      
        if ind_out == None:
            nvars = len(self.est_list)
            ind_out = np.arange(nvars)
        zmean = []
        zvar = []
        cost = 0
        for i, est in enumerate(self.est_list):
            if return_cost:
                zmeani, zvari, ci = est.est_init(return_cost)
                cost += ci
            else:
                zmeani, zvari = est.est_init(return_cost)
            if i in ind_out:
                zmean.append(zmeani)
                zvar.append(zvari)
        if return_cost:
            return zmean, zvar, cost
        else:
            return zmean, zvar
            
    def est(self,r,rvar,return_cost=False,ind_out=None,\
        avg_var_cost=True):
        """
        Estimation function
        
        The proximal estimation function as 
        described in the base class :class:`vampyre.estim.base.Estim`
                
        :param r: Proximal mean
        :param rvar: Proximal variance
        :param Boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        
        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior
            mean, variance and optional cost.
        """        
        if ind_out == None:
            nvars = len(self.est_list)
            ind_out = np.arange(nvars)

        zhat = []
        zhatvar = []
        cost = 0
        for i, est in enumerate(self.est_list):
            ri = r[i]
            rvari = rvar[i]
            if return_cost:
                zhati, zhatvari, ci = est.est(ri,rvari,return_cost)
                cost += ci
            else:
                zhati, zhatvari = est.est(ri,rvari,return_cost)
            if i in ind_out:
                zhat.append(zhati)
                zhatvar.append(zhatvari)
            
        if return_cost:
            return zhat, zhatvar, cost
        else:
            return zhat, zhatvar
     
            
                
            