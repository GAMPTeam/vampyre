"""
stack.py:  Concantenated or "stacked" estimators
"""
from __future__ import division
from __future__ import print_function

# Import individual classes and methods from the current subpackage
from vampyre.estim.base import Estim

class StackEstim(Estim):
    """
    A stacked estimator
    
    This is an estimator for a list of variables 
    :math:`z=[z_0,\\ldots,z_{K-1}]`.  
    
    :param est_list:  List of estimators for each variable
    """
    def __init__(self,est_list):
        Estim.__init__(self)
        self.est_list = est_list
        
        # Cost is available if all estimators have a cost available
        cost_avail = True
        for est in est_list:
            cost_avail = cost_avail and est.cost_avail
        self.cost_avail = cost_avail
        
    def est_init(self, return_cost=False):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.
              
        :param boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """      
        zmean = []
        zvar = []
        cost = 0
        for est in self.est_list:
            if return_cost:
                zmeani, zvari, ci = est.est_init(return_cost)
                cost += ci
            else:
                zmeani, zvari = est.est_init(return_cost)
            zmean.append(zmeani)
            zvar.append(zvari)
        if return_cost:
            return zmean, zvar, cost
        else:
            return zmean, zvar
            
    def est(self,r,rvar,return_cost=False):
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
        zhat = []
        zhatvar = []
        cost = 0
        i = 0
        for est in self.est_list:
            ri = r[i]
            rvari = rvar[i]
            if return_cost:
                zhati, zhatvari, ci = est.est(ri,rvari,return_cost)
                cost += ci
            else:
                zhati, zhatvari = est.est(ri,rvari,return_cost)
            zhat.append(zhati)
            zhatvar.append(zhatvari)
            i += 1
            
        if return_cost:
            return zhat, zhatvar, cost
        else:
            return zhat, zhatvar
     
            
                
            