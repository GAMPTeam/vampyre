import numpy as np
import vampyre.common as common
from vampyre.estim import ReLUEstim

def test_ReLUEstim(zshape=(1000,10),tol=0.15,verbose=False):
    """
    ReLUEstim unit test.
    
    In this test, a matrix :math:`z_0=r_0+w_0` is generated with iid Gaussian 
    components, :math:z_1 = \\max(0,z_0)` is the ReLU ouptut of :math:`z_0`
    and :math:`r_1 = z_1 + w_1`.  The ReLU estimator is then used to
    estimate :math:`z_i` from :math:`r_i`.  The test passes if the 
    predicted variance matches the true variance with a specified tolerance.
            
    :param zshape: shape of :math:`z_0` and :math:`z_1`.
    :param tol:  Tolerance for test test to pass
    :param Boolean verbose:  prints results.  
    """
     
    # Get number of columns
    ns = zshape[1]
        
    # Set random parameters
    rvar0 = np.power(10,np.random.uniform(-2,1,ns))
    rvar1 = np.power(10,np.random.uniform(-2,1,ns))
        
    # Construct random input
    r0 = np.random.normal(0,1,zshape) 
    z0 = r0 + np.random.normal(0,1,zshape)*np.sqrt(rvar0)[None,:]
    
    # Perform ReLU and add noise
    z1 = np.maximum(z0,0)
    r1 = z1 + np.random.normal(0,1,zshape)*np.sqrt(rvar1)[None,:]
    
    # Construct the estimator
    relu = ReLUEstim(shape=zshape)
    
    # Repeath the variances
    r = [r0,r1]
    rvar = [rvar0,rvar1]
    zhat, zhatvar, cost = relu.est(r,rvar,return_cost=True)
    
    # Unpack the estimates
    zhat0, zhat1 = zhat
    zhatvar0, zhatvar1 = zhatvar
    
    # Compute the true error and compare to the 
    zerr0 = np.mean((zhat0-z0)**2,axis=0)
    zerr1 = np.mean((zhat1-z1)**2,axis=0)
    
    # Compute average difference
    diff0 = np.mean(np.maximum(zhatvar0/zerr0,zerr0/zhatvar0))-1
    diff1 = np.mean(np.maximum(zhatvar1/zerr1,zerr1/zhatvar1))-1
    
    # Check if fails
    fail = (diff0 > tol) or (diff1 > tol)    
    if verbose or fail:
        print("")
        print("z0 err: act: {0:12.4e} pred: {1:12.4e} diff: {2:12.4e}".format(\
            np.mean(zerr0),np.mean(zhatvar0),diff0))
        print("z1 err: act: {0:12.4e} pred: {1:12.4e} diff: {2:12.4e}".format(\
            np.mean(zerr1),np.mean(zhatvar1),diff1)) 
    if fail:
        raise common.TestException("Posterior predicted variance does not match "+\
            "actual variance within tolerance")
    