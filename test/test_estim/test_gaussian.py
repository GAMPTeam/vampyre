import sys
sys.path.append('.')
import vampyre as vp
import numpy as np
from vampyre.estim import GaussEst

def test_gauss(zshape=(1000,10), verbose=False, tol=0.1):
    """
    Unit test for the Gaussian estimator class :class:`GaussEst`.
    
    The test works by creating synthetic Gaussian variables :math:`z`, and 
    measurements 
    
    :math:`r = z + w,  w \sim {\mathcal N}(0,\\tau_r)`
    
    Then, the :func:`Gaussian.est` and :func:`Gaussian.est_init` methods 
    are called to see if :math:`z` with the expected variance.
                
    :param zshape: Shape of :param:`z`.  The parameter can be an 
       arbitrary :code:`ndarray'.
    :param Boolean verbose:  Flag indicating if results are to be printed
    :param tol:  Tolerance for raising an error
    :param Boolean raise_exception: raise an :class:`TestException` if 
       test fails.
    """    

    # Generate synthetic data with random parameters
    zvar =  np.random.uniform(0,1,1)[0]
    rvar = np.random.uniform(0,1,1)[0]
    zmean = np.random.normal(0,1,1)[0]
    z = zmean + np.random.normal(0,np.sqrt(zvar),zshape)
    r = z + np.random.normal(0,np.sqrt(rvar),zshape)
    
    # Construct estimator
    est = GaussEst(zmean,zvar,zshape,var_axes='all')

    # Inital estimate
    zmean1, zvar1 = est.est_init()
    zerr = np.mean(np.abs(z-zmean1)**2)
    if verbose:
        print("Initial:      True: {0:f} Est:{1:f}".format(zerr,zvar1))
    if (np.abs(zerr-zvar1) > tol*np.abs(zerr)):
        raise TestException("Initial estimate Gaussian error "+ 
           "does not match predicted value")
    
    # Posterior estimate
    zhat, zhatvar, cost = est.est(r,rvar,return_cost=True)
    zerr = np.mean(np.abs(z-zhat)**2)
    if verbose:
        print("Posterior:    True: {0:f} Est:{1:f}".format(zerr,zhatvar))
    if (np.abs(zerr-zhatvar) > tol*np.abs(zhatvar)):
        raise TestException("Posterior estimate Gaussian error "+ 
           "does not match predicted value") 