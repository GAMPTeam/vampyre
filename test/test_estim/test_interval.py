import numpy as np
import vampyre.common as common
from vampyre.estim import HardThreshEst
from vampyre.estim.interval import gauss_integral
from scipy.integrate import quad

def test_HardThreshEst(zshape=(1000,10), verbose=False, rvar=None, tol=0.1):
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

def test_gauss_integral(v=None,mu=2,var=0.3,ns=100,verbose=False,tol=1e-8):
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