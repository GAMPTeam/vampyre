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
from vampyre.estim.base import Estim
from vampyre.estim.gaussian import GaussEst


class MixEst(Estim):
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
    def __init__(self, est_list, w):
        Estim.__init__(self)
        self.est_list = est_list
        self.w = w
        self.shape = est_list[0].shape
        self.var_axes = est_list[0].var_axes
        
        # Check that all estimators have cost available
        for est in est_list:
            if not est.cost_avail:
                raise common.VpException(\
                    "Estimators in a mixture distribution"+\
                    "must have cost_avail==True")
        self.cost_avail = True
                                 
    def est_init(self, return_cost=False):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.
        
        :param Boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """       
        
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
    
                    
    def est(self,r,rvar,return_cost=False):
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
        
        # Save the probability
        self.prob = p_list
        
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
            
def mix_test(zshape=(1000,10), verbose=False, tol=0.1, raise_exception=True):
    """
    Unit test for the :class:`MixEst` class
    
    The test works by creating synthetic Gaussian mixture
    variables :math:`z`, and measurements 
    
    :math:`r = z + w,  w \sim {\mathcal N}(0,\\tau_r)`
    
    Then, the estimateion methods are called to see if 
    the measured error variance matches the expected value.
            
    :param zshape: shape of :math:`z`
    :param Boolean verbose:  prints results.  
    :param tol:  error tolerance to consider test as passed
    :param Boolean raise_exception:  Raise an error if test fails. 
       The exception can be caught by the test_suite dispatcher.
    """    

    # Generate random components    
    ncomp=3
    nz = np.prod(zshape)     
    zvar = np.power(10,np.random.uniform(-2,1,ncomp))
    rvar = np.power(10,np.random.uniform(-2,1,ncomp))[0]
    zmean = np.random.normal(0,1,ncomp)
    
    # Generate random probabilities
    p = np.random.rand(ncomp)
    p = p / np.sum(p)
    
    # Create a set of estimators
    est_list = []
    zcomp = np.zeros((nz,ncomp))
    for i in range(ncomp):
        est = GaussEst(zmean[i], zvar[i], zshape, var_axes='all')
        est_list.append(est)
        
        zcomp[:,i] = np.random.normal(zmean[i],np.sqrt(zvar[i]),nz)
        
    # Compute the component selections
    u = np.random.choice(range(ncomp),p=p,size=nz)
    z = np.zeros(nz)
    for i in range(nz):
        z[i] = zcomp[i,u[i]]
    z = np.reshape(z,zshape)
    
    # Add noise
    r = z + np.random.normal(0,np.sqrt(rvar),zshape)    
    
    # Construct the estimator
    est = MixEst(est_list, w=p)     
    
    # Inital estimate
    zmean1, zvar1 = est.est_init()
    zerr1 = np.mean(np.abs(z-zmean1)**2)
    if verbose:
        print("Initial:    True: {0:f} Est:{1:f}".format(zerr1,zvar1))
    if (np.abs(zerr1-zvar1) > tol*np.abs(zerr1)) and raise_exception:
        raise common.TestException("Initial estimate GMM error "+ 
           " does not match predicted value")
    
    # Posterior estimate
    zhat, zhatvar, cost = est.est(r,rvar,return_cost=True)
    zerr = np.mean(np.abs(z-zhat)**2)
    if verbose:
        print("Posterior:  True: {0:f} Est:{1:f}".format(zerr,zhatvar))
    if (np.abs(zerr-zhatvar) > tol*np.abs(zerr)) and raise_exception:
        raise common.TestException("Posterior estimate GMM error "+ 
           " does not match predicted value")

