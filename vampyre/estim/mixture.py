"""
mixture.py:  Classes for estimators for mixture distributions
"""
from __future__ import division
from __future__ import print_function


# Import general packages
import numpy as np
import copy

# Import other sub-packages
import vampyre.common as common

# Import methods and classes from the same sub-package
from vampyre.estim.base import BaseEst


class MixEst(BaseEst):
    """ Mixture estimator class
    
    Given a list of estimators where :code:`est_list[i]`, 
    corresponds to a penalty, :math:`f_i(z)`, this class
    creates a new estimator corresponding to the penalty,
    :math:`f(z)=-\\ln( \\sum_i w_i \\exp(-f_i(z))`, 
    for a set of weights :math:`w_i`.
    
    :note:  If :math:`\\int e^{-f_i(z)}dz=1` for all :math:`i` and
        and :math:`\\sum_i w_i=1`, then :math:`w_i` can be interpreted
        as the probability of the :math:`i`-th component.
        
    :note:  The variables must be scalar and the estimators 
        must have attributes :code:`shape` and :code:`var_axes`.
        Also, the estimator's :code:`est_init` and :code:`est`
        methods must implement a :code:`avg_var_cost` option to
        disable averaging of the averaging the variance and cost.
    
        
    :param est_list:  list of estimators 
    :param w: weight for the components
    """    
    def __init__(self, est_list, w, name=None):
        
        self.est_list = est_list
        self.w = w
        
        shape = est_list[0].shape
        var_axes = est_list[0].var_axes
        dtype = est_list[0].dtype
        
        # Check that all estimators have cost available
        for est in est_list:
            if est.shape != shape:
                raise common.VpException('Estimators must have the same shape')
            if est.var_axes != var_axes:
                raise common.VpException('Estimators must have the same var_axes')                
            if not est.cost_avail:
                raise common.VpException(\
                    "Estimators in a mixture distribution"+\
                    "must have cost_avail==True")
        BaseEst.__init__(self,shape=shape,var_axes=var_axes,dtype=dtype,\
            name=name, type_name='Mixture', nvars=1, cost_avail=True)
                                 
    def est_init(self, return_cost=False, ind_out=None,\
        avg_var_cost=True):
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
        if (ind_out != [0]) and (ind_out != None):
            raise ValueError("ind_out must be either [0] or None")
        if not avg_var_cost:
            raise ValueError("disabling variance averaging not supported for MixEst")

        # Get the mean and variance of each component
        zmean_list = []
        zvar_list = []
        cost_list = []
        for i,est in enumerate(self.est_list):
            zmeani, zvari, ci = est.est_init(return_cost=True,avg_var_cost=False)
            zmean_list.append(zmeani)
            zvar_list.append(zvari)
            cost_list.append(ci)
            
        return self._comp_est(zmean_list,zvar_list,cost_list,return_cost)
    
                    
    def est(self,r,rvar,return_cost=False,ind_out=None,\
        avg_var_cost=True):
        """
        Estimation function
        
        The proximal estimation function as 
        described in the base class :class:`vampyre.estim.base.Estim`
                
        :param r: Proximal mean
        :param rvar: Proximal variance
        :param boolean return_cost:  Flag indicating if :code:`cost` is to be returned
        
        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior 
        mean, variance and optional cost.
        """
        
        # Check parameters
        if (ind_out != [0]) and (ind_out != None):
            raise ValueError("ind_out must be either [0] or None")
        if not avg_var_cost:
            raise ValueError("disabling variance averaging not supported for MixEst")

        # Get the mean and variance of each component
        zmean_list = []
        zvar_list = []
        cost_list = []
        for i,est in enumerate(self.est_list):
            zmeani, zvari, ci = \
               est.est(r,rvar,return_cost=True,avg_var_cost=False)
            zmean_list.append(zmeani)
            zvar_list.append(zvari)
            cost_list.append(ci)
          
        return self._comp_est(zmean_list,zvar_list,cost_list,return_cost)
        
    def _comp_est(self,zmean_list,zvar_list,cost_list,return_cost):
        """
        Computes the estimates given mean, variance and costs on each 
        component.  
        
        :param zmean_list: list of mean values for each component.
           The values are stored as a list where each element is an
           array of shape :code:`self.shape`.
        :param zvar_list:  list of variance values
        :param cost_list:  list of cost values
        :param return_cost:  Flag indicating if to return the cost.        
        """
            
        # Find the minimum cost.  This will be subtracted from all the costs
        # to prevent overflow when taking an exponential
        ncomp = len(self.w)
        cmin = copy.deepcopy(cost_list[0])
        for i in range(1,ncomp):
            cmin = np.minimum(cmin, cost_list[i])
            
        # Compute p_list[i] \prop w[i]*exp(-cost_list[i]),
        # which represents the probabilities for each component.
        p_list = []
        psum = np.zeros(self.shape)
        for i in range(ncomp):
            pi = self.w[i]*np.exp(-cost_list[i] + cmin)
            psum += pi
            p_list.append(pi)
        cost = np.sum(-np.log(psum) + cmin)
        for i in range(ncomp):
            p_list[i] /= psum        
        
        # Save the probability, and conditional means and variances
        self.prob = p_list
        self.zmean_list = zmean_list
        self.zvar_list = zvar_list
        
        # Compute prior mean and variance
        zmean = np.zeros(self.shape)
        zsq = np.zeros(self.shape)
        for i in range(ncomp):
            zmean += p_list[i]*zmean_list[i]
            zsq   += p_list[i]*(zvar_list[i] + np.abs(zmean_list[i])**2)
        zvar = zsq - np.abs(zmean)**2        
        zvar = np.mean(zvar, axis=self.var_axes)
         
        if return_cost:
            return zmean, zvar, cost
        else:
            return zmean, zvar

