"""
interval.py:  Estimators for outputs on intervals
"""
from __future__ import division

import numpy as np
import scipy.special
from scipy.integrate import quad

# Import other subpackages in vampyre
import vampyre.common as common

# Import individual classes and methods from the current subpackage
from vampyre.estim.base import Estim

class HardThreshEst(Estim):
    """
    Esitmator for a hard threshold
    
    This esitmator corresponds to a binary measurement :math:`y=0,1` where
    :math:`y=1` if :math:`z > t` and :math:`y=0` else.  The value :math:`t`
    is a threshold.  
    
    To improve the numerical stability of the estimator for outliers, the
    estimator has a parameter :math:`\\lambda` such that :math:`y` changes
    sign with probality :math:`\\lambda`.  This should be set to a small, 
    but positive number.
    
    :param y:  Binary output 
    :param zrep_axes:  The axes on which the input variance is repeated.
        Default is 'all'.
    :param shape: Shape of :math:`y` and :math:`z`
    :param thresh: Threshold (default = 0)
    :param perr:  error probability (default = 1e-6)
    :param var_init:  initial variance.  This should be large (default=100)
    """    
    def __init__(self,y,shape,zrep_axes=(0,),thresh=0,perr=1e-6,\
                 var_init=np.Inf):
        
        Estim.__init__(self)
        self.y = y
        self.shape = shape
        self.thresh = thresh
        self.perr = perr
        self.cost_avail = True
        self.var_init = var_init
       
        # Set the repetition axes
        ndim = len(self.shape)
        if zrep_axes == 'all':
            zrep_axes = tuple(range(ndim))
        self.zrep_axes = zrep_axes        
                                  
        
    def est_init(self, return_cost=False):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.
        
        The initial estimate is technically unbounded so we set a large
        variance.
        
        :param boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """        
        zmean = np.zeros(self.shape)
        ndim = len(self.shape)
        zvar_ind = [i for i in range(ndim) if i not in self.zrep_axes]
        if zvar_ind == []:
            zvar = self.var_init
        else:
            zvar_shape = np.array(self.shape)[zvar_ind]
            zvar_shape = tuple(zvar_shape)        
            zvar = self.var_init*np.ones(zvar_shape)
        cost = 0
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
        
        # Repeat the variance and reshape r and rvar to 1D vectors
        rvar1 = common.repeat_axes(rvar, self.shape, self.zrep_axes)
        r1 = r.ravel()
        rvar1 = rvar1.ravel()
        y1 = self.y.ravel()
                
        # Compute the values for Ai 
        #   A0i = \int_{-\intfy}^thresh z^i exp(-(z-r)^2/(2*rvar))
        #   A1i = \int_thresh^\infty z^i exp(-(z-r)^2/(2*rvar))
        rsig = np.sqrt(rvar1)
        A00,A01,A02 = gauss_integral(-np.Inf,self.thresh,r1,rvar1)        
        A10 = rsig-A00
        A11 = rsig*r1-A01
        A12 = rsig*(rvar1+r1**2)-A02
        
        # Compute probability y==1 before flipping
        py1 = y1*(1-self.perr) + (1-y1)*self.perr        
        
        # Set Ai = A0i for y==0 and Ai=A1i for y==1
        A0 = A10*py1 + A00*(1-py1)
        A1 = A11*py1 + A01*(1-py1)
        A2 = A12*py1 + A02*(1-py1)
        
        # Compute posterior mean and variance
        zhat = A1/A0
        zhatvar = A2/A0-(zhat**2)
        cost = -np.sum(np.log(A0))
        
        # Reshape and average values
        zhat = np.reshape(zhat,self.shape)
        zhatvar = np.reshape(zhatvar,self.shape)
        zhatvar = np.mean(zhatvar, axis=self.zrep_axes)
        
        if return_cost:
            return zhat, zhatvar, cost
        else:
            return zhat, zhatvar
                
                      
def hard_thresh_test(zshape=(1000,10), verbose=False, rvar=None, tol=0.1):
    """
    Unit test for the :class:`HardThreshEst` class
    
    The test works by creating synthetic distribution, creating an i.i.d.
    matrix :math:`z` with Gaussian components and then taking quantized
    values :math:`y = (z > t)` where `t` is a random threshold    
    Then, the estimation methods are called to see if 
    the measured error variance matches the expected value.
            
    :param zshape: shape of :math:`z`
    :param Boolean verbose:  prints results.  
    :parma rvar:  Gaussian variance.  :code:`None` will generate a random
       value
    :param tol:  Tolerance for test test to pass
    """    
    
    # Generate random parameters
    if rvar == None:
        rvar = 10**(np.random.uniform(-1,1,1))[0]
    
    # Generate data
    r = np.random.normal(0,1,zshape)
    z = r + np.random.normal(0,np.sqrt(rvar),zshape)
    thresh = 0 #np.random.uniform(-1,1,1)[0]   
    y = (z > thresh)
    
    # Create estimator
    est = HardThreshEst(y,shape=zshape,thresh=thresh,zrep_axes='all')
    
    # Run the initial estimate.  In this case, we just check that the 
    # dimensions match
    zmean, zvar, cost = est.est_init(return_cost=True)
    if not (zmean.shape == zshape) or not np.isscalar(zvar):
        raise common.TestException("Initial shapes are not correct")
                    
    # Get posterior estimate
    zhat, zhatvar, cost = est.est(r,rvar,return_cost=True)
    
    # Measure error
    zerr = np.mean(np.abs(zhat-z)**2)
    if verbose:
        print("err: true: {0:12.4e} est: {1:12.4e}".format(zerr,zhatvar) )
    if (np.abs(zerr-zhatvar) > tol*np.abs(zerr)):
        raise common.TestException("Posterior estimate for error "+ 
           "does not match predicted value")


def gauss_integral(a,b,mu,var):
    """
    Computes Gaussian integral on a single interval
    
       z[i] = 1/\sqrt(2*pi)\int_a^b x^i exp(-(x-mu)**2/(2*tau))dx
       
    :param a: lower limit of the integral
    :param b: upper limit of the integral
    :param mu:  mean of the Gaussian
    :param var:  variance of the Gaussian
    
    :returns:
    :param z0,z1,z2:  The integral values
    """
    
    """
    The computation is made by first taking a substitution
    
       u = (x-mu)/sig  dx = sig*du
       alpha = (a-mu)/sig, beta = (b-mu)/sig
       
       z[i] = sig*\int_alpha^beta (mu + sig*u)^i exp(-u^2/2)du
       
       Then, we compute
       w[i] = \int_alpha^\beta u^i exp(-u^2/2)du
    """
    
    # Get standard deviation
    sig = np.sqrt(var)
        
    if (b == np.Inf):
        # Integral from [a,infty]
        alpha = (a-mu)/sig
        f = 1/(np.sqrt(2*np.pi))*np.exp(-alpha**2/2)
        Q = 0.5*scipy.special.erfc(alpha/np.sqrt(2))        
        w0 = Q
        w1 = f
        w2 = Q+alpha*f
    elif (a == -np.Inf):
        # Integral from [-infty,b]
        beta = (b-mu)/sig
        f = 1/(np.sqrt(2*np.pi))*np.exp(-beta**2/2)
        F = 0.5*(1+scipy.special.erf(beta/np.sqrt(2)))
        w0 = F
        w1 = -f
        w2 = F-beta*f        
    else:
        raise Exception("Only single sided distributions handled for now")
      
    # Convert back to z
    z0 = sig*w0
    z1 = sig*(mu*w0 + sig*w1)
    z2 = sig*((mu**2)*w0 + 2*mu*sig*w1 + var*w2)    
        
    return z0,z1,z2
    

def gauss_integral_test(v=None,mu=2,var=0.3,ns=100,verbose=False,tol=1e-8):
    """
    Unit test for the gauss inetegral function
    
    The gaussian integration is tested by computing the integral
    in two intervals: :math:`[-\infty,v]` and :math:`[v,\infty]` for
    for some value :math:`v`.  The integral is compared to the
    quadrature integration.
    
    :param v:  integral limit (:code:`None` indicates parameters are
       generated randomly
    :param mu:  Gaussian mean
    :param var:  Gaussian variance
    :param ns:  number of samples
    :param Boolean verbose:  Print results
    :param tol:  Tolerance to consider test as passed    
    """
    
    # Randomly generate values
    if v==None:
        v  = np.random.randn(1)
        mu = 2*np.random.randn(ns)
        var = 10**(np.random.uniform(-1,1,ns))
        
    # Lower integral test
    for it in range(2):
        if it == 0:
            a = -np.Inf
            b = v
            tstr = 'lower'
        else:
            a = v
            b = np.Inf
            tstr = 'upper'
        
        err = 0
        zest = gauss_integral(a,b,mu,var)                    
        for j in range(3):
            zquad = np.zeros(ns)
            for i in range(ns):             
                f = lambda u: 1/np.sqrt(2*np.pi)*(u**j)*np.exp( -(u-mu[i])**2/(2*var[i]))
                zquad[i] = quad(f, a, b)[0]
        
            err += np.mean( np.abs(zquad - zest[j]) )
        if verbose or (err > tol):
            print("{0:s} integral error:  {1:12.4e}".format(tstr,err))
        if err > tol:
            raise common.TestException(\
                "Gaussian integral does not match quadrature")