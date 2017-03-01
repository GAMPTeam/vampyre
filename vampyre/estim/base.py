# -*- coding: utf-8 -*-
"""
base.py:  Defines base class for estimators
"""
from __future__ import division

class Estim(object):
    """Base class for estimators
    
    An estimator is most commonly derived on a penalty function :math:`f(z)`.
    Corresponding to this penalty function is a 
    probability density, 
    
    :math:`p(z)=(1/Z)\\exp(-f(z))`.  
    
    The estimatortypically does not directly implement the penalty
    :math:`f(z)`, but rather implements MAP and/or MMSE estimators 
    based on the penalty as described below.
    The estimator can also be derived without an explicit penalty and simply
    implement the estimator methods.
    
    The methods in this base class are mostly abstract.  Estimators should 
    derive from this base class.
    """    
    def __init__(self):
        # Can compute the cost
        self.cost_avail = False
    
    def est_init(self, return_cost=False):
        """ Initial estimate.  
        
        Given the penalty function, :math:`f(z)`, the methods computes
        initial estimates as follows.
        For MAP estimation, this method should return:
        
        * :math:`\\hat{z} = \\arg \\min_z f(z)`
        * :math:`\\tau_z = 1/\\langle f''(\\hat{z}) \\rangle`        
        * :math:`c = \\min_z f(z)`. 
            
        For MMSE estimation, this should return:
        
        * :math:`\\hat{z} = E(z)` 
        * :math:`\\tau_z = \\mathrm{var}(z)`
        * :math:`c = -\\ln Z`, where :math:`Z=\int e^{-f(z)}dz`.  This can 
          also be computed by :math:`c = E[f|p] - H(p)` where :math:`p` is the 
          density :math:`p(z) = \\exp(-f(z))`.
            
       
        The parameters are:
        
        :param boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned

        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior
            mean, variance and optional cost as defined above.        
        """
        raise NotImplementedError()       
        
    def est(self,r,rvar,return_cost=False):
        """ Proximal estimator
        
        Given the penalty function, :math:`f(z)`, define the augmented penalty:        
        
        :math:`f_a(z) = f(z) + (1/2\\tau_r)|z-r|^2`
            
        and the associated augmented density 
        :math:`p(z|r,\\tau_r) = (1/Z)exp(-f_a(z))`.  This method
        then returns MAP or MMSE estimates based on the penalty function.
        
        Spicifically, for MAP estimation, this should return:
        
        * :math:`\\hat{z} = \\arg \\max_z p(z|r,\\tau_r) = \\arg \\min_z f_a(z)`
        * :math:`\\tau_z = 1/<f_a''(\\hat{z})>`        
        * :math:`c = \\min_z f_a(z)`. 
            
        For MMSE estimation, this should return:
        
        * :math:`\\hat{z} = E(z|r,\\tau_r)`
        * :math:`\\tau_z = \\mathrm{var}(z|r,\\tau_r)`
        * :math:`c = -\\ln Z_a`, where :math:`Z_a` is the partition function, 
          :math:`Z_a=\int e^{-f_a(z)}dz`.  This can also be computed by 
          :math:`c = E[f|r,\\tau_r] - H(p)` where :math:`p` is the 
          conditional density :math:`p(z|r,\\tau_r)` above.
       
        The parameters are:
        
        :param r: Proximal mean
        :param rvar: Proximal variance
        :param boolean return_cost:  Flag indicating if :code:`cost` 
            is to be returned

        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior
            mean, variance and optional cost as defined above.        
        """
        raise NotImplementedError()
           
    
        