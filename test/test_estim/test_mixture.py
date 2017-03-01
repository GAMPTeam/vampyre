import numpy as np
from vampyre.estim import GaussEst, MixEst
import vampyre.common as common

def test_mixest(zshape=(1000,10), verbose=False, tol=0.1, raise_exception=True):
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

