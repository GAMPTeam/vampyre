""" Tests for the :class:`DiscreteEst` class
"""    
import sys
sys.path.append('..')
import vampyre as vp
import numpy as np
import vampyre.common as common
from vampyre.estim import DiscreteEst

def test_discrete(zshape=(1000,10), verbose=False, nvals=3,\
             tol_init=1e-3, tol_est=0.2):
    """ This test works by creating synthetic distribution, creating an i.i.d.
    matrix :math:`z` with components from that distribution and then 
    Gaussian measurements 
    
    :math:`r = z + w,  w \sim {\mathcal N}(0,\\tau_r)`
    
    Then, the estimateion methods are called to see if 
    the measured error variance matches the expected value.
            
    :param zshape: shape of :math:`z`
    :param Boolean verbose:  prints results.  
    :param tol_init:  tolerance on initial estimate for test to be considered
       a pass.  This tolerance should be very low.
    :param tol_est:  Error tolerance on the esitmation error.  This should
       be much higher since the Monte Carlo simulations take a large number
       of samples to converge.
    :param nvals:  number of values in the discrete distribution
    """    
    # Generate a random discrete distribution
    zval = np.random.randn(nvals)
    pz = np.random.rand(nvals)
    pz = pz/sum(pz)
    
    # Noise variance
    #rvar = np.power(10,np.random.uniform(-2,1,1))[0]
    rvar = 0.1
    
    # Generate random data
    z = np.random.choice(zval,zshape,p=pz)
    r = z + np.random.normal(0,np.sqrt(rvar),zshape)
    
    # Create estimator
    est = DiscreteEst(zval, pz, zshape, var_axes='all')
    
    # Run the initial estimate
    zmean, zvar, cost = est.est_init(return_cost=True)
    
    # Compute the true expected mean
    zmean0 = pz.dot(zval)
    if np.abs(zmean0 -np.mean(zmean)) > tol_init:
        raise common.TestException("Initial mean does not match expected value")
        
    # Compute the true expected variance
    zvar0 = pz.dot(np.abs(zval-zmean0)**2)
    if np.abs(zvar0 -np.mean(zvar)) > tol_init:
        raise common.TestException(\
           "Initial variance does not match expected value")
    
    # Get posterior estimate
    zhat, zhatvar, cost = est.est(r,rvar,return_cost=True)
    
    # Measure error
    zerr = np.mean(np.abs(zhat-z)**2)
    fail = (np.abs(zerr-zhatvar) > tol_est*np.abs(zerr))
    if verbose or fail:
        print("err: true: {0:12.4e} est: {1:12.4e}".format(zerr,zhatvar) )
    if fail:
        raise common.TestException("Posterior estimate discrete error "+ 
           "does not match predicted value")

