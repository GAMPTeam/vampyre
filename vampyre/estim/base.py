# -*- coding: utf-8 -*-
"""
base.py:  Defines base class for estimators
"""
from __future__ import division

import numpy as np

class BaseEst(object):
    """
    Static variables
    """
    ind_name = 0    # Unique ID for each estimator used in naming
    
    def __init__(self, shape=None, var_axes=(0,), dtype=np.float64, name=None,\
        type_name='Estim', nvars=1, cost_avail=False):
        """
        Base class for estimators.
    
        An estimator is most commonly derived on a penalty function :math:`f(z)`.
        Corresponding to this penalty function is a probability density, 
        
        :math:`p(z)=(1/Z)\\exp(-f(z))`.  
        
        The estimator typically does not directly implement the penalty
        :math:`f(z)`, but rather implements MAP and/or MMSE estimators 
        based on the penalty as described below.
        The estimator can also be derived without an explicit penalty and simply
        implement the estimator methods.
        
        The class also supports estimators for multiple variables where the 
        penalty function is of the form `f(z[0],...,z[nvars-1])`.  
        
        The methods in this base class are mostly abstract.  Estimators should 
        derive from this base class.  
        
        :shape:  Shape of the variable tensor.  If `nvar>1` then this should be 
            a list of shapes.
        :var_axes:  The axes over which the variances are to be averaged.
        :dtype:  Data type (default `np.double`).  If `nvar>1`, 
            this should be a list of data types.
        :param name:  String name of the estimator 
        :param type_name:  String name of the estimator type
        :param nvars:  Number of variable nodes, `nvars`, connected to the estimator.
        :param cost_avail:  Flag indicating if the estimator can compute the cost.
        """    

        if np.isscalar(shape):
            shape = (shape,)
        self.shape = shape
        self.dtype = dtype
        self.type_name = type_name
        if name is None:
            self.name = 'Est_' + str(BaseEst.ind_name)
            BaseEst.ind_name += 1
        else:
            self.name = name
        self.nvars = nvars
        self.cost_avail = cost_avail        
        if var_axes == 'all':
            ndim = len(self.shape)
            self.var_axes = tuple(range(ndim))
        else:
            self.var_axes = var_axes
    
    def est_init(self, return_cost=False, ind_out=None, avg_var_cost=True):
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
        :param ind_out:  When `nvars>1`, this is the list of the indices of 
            all variables thatare to be returned.  Items should be organized as a list.  If 
            `ind_out==None`, then it returns estimates for all variables.            
        :param avg_var_cost:  If variance and cost are to be averaged per
            element.  This is normally set to :code:`True`, but is set
            to :code:`False` when using mixture distributions.  

        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior
            mean, variance and optional cost as defined above.        
        """
        raise NotImplementedError()       
        
    def est(self,r,rvar,return_cost=False,ind_out=None):
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
        :param ind_out:  When `nvars>1`, this is the list of the indices of 
            all variables thatare to be returned.  Items should be organized as a list.  If 
            `ind_out==None`, then it returns estimates for all variables.
        :param boolean return_cost:  Flag indicating if :code:`cost` 
            is to be returned
        

        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior
            mean, variance and optional cost as defined above.        
        """
        raise NotImplementedError()
           
    
    def __str__(self):
        string = str(self.type_name) + ', name: ' + str(self.name) + ', '\
                  + 'shape: ' + str(self.shape) + ', type:' + str(self.dtype)
        return string
