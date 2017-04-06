import numpy as np
import vampyre.common as common
from vampyre.estim import LinEstim
import vampyre.trans as trans

def lin_est(zshape=(500,10),Ashape=(1000,500),verbose=False,tol=0.1):
    """
    Unit test for the linear estimator class
    
    The test is performed by generating random data 
    
    :math:`y=Az+w,  z \\sim {\\mathcal N}(r, \\tau_r I), 
       w \\sim {\\mathcal N}(0, \\tau_w I)`
       
    Then the method estimates :math:`z` from :math:`y` 
    and compares the expected and measured errors.
    
    :param zshape:  shape of :math:`z`
    :param Ashape:  shape of :A:`z`.  This must be consistent with 
       :code:`zshape`.
    :param Boolenan verbose:  print results
    :param tol:  error tolerance above which test is considered
       to fail.    
    """            
     
    # Generate random parameters
    rvar = 10**(np.random.uniform(-1,1,1))[0]
    wvar = 10**(np.random.uniform(-1,1,1))[0]    
        
    # Generate random matrix
    A = np.random.normal(0,1,Ashape)/np.sqrt(Ashape[1])
    Aop = trans.MatrixLT(A, zshape)
    yshape = Aop.shape1
    
    # Add noise on input and output
    r = np.random.normal(0,1,zshape) 
    z = r + np.random.normal(0,np.sqrt(rvar),zshape)
    y = A.dot(z) + np.random.normal(0,np.sqrt(wvar),yshape)
    
    # Construct the linear estimator
    est = LinEstim(Aop,y,wvar,zrep_axes='all')
    
    # Perform the initial estimate.  This is just run to make sure it
    # doesn't crash
    zhat, zhatvar, cost = est.est_init(return_cost=True)
    if (zhat.shape != r.shape) or (zhatvar.shape != wvar):
        raise common.TestException(\
           "est_init does not produce the correct shape")            
    
    # Posterior estimate
    zhat, zhatvar, cost = est.est(r,rvar,return_cost=True)
    zerr = np.mean(np.abs(z-zhat)**2)
    fail = (np.abs(zerr-zhatvar) > tol*np.abs(zhatvar))
    if verbose or fail:
        print("\nPosterior:    True: {0:f} Est:{1:f}".format(zerr,zhatvar))
    if fail:
       raise common.TestException("Posterior estimate Gaussian error "+ 
          " does not match predicted value")      
    
# def lin_test_mult():
#     """
#     Unit tests for the linear estimator class
    
#     This calls :func:`lin_test` with multiple different paramter values
#     """
#     lin_test(zshape=(500,10),Ashape=(1000,500))
#     lin_test(zshape=(500,),Ashape=(1000,500),tol=0.5)
#     lin_test(zshape=(500,10),Ashape=(250,500))


def test_linest_z500x10_A1000x500():
    lin_est(zshape=(500,10),Ashape=(1000,500))

def test_linest_z500x1_A1000x500():    
    lin_est(zshape=(500,),Ashape=(1000,500),tol=0.5)

def test_linest_z500x10_A250x500():
    lin_est(zshape=(500,10),Ashape=(250,500))