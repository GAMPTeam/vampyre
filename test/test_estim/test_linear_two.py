import numpy as np
import vampyre.common as common
from vampyre.estim import LinEstimTwo
import vampyre.trans as trans

def test_LinTwoEst(nz0=100,nz1=200,ns=10,map_est=False,verbose=False,tol=1e-8):
    """
    Unit test for the linear estimator class
    
    The test is performed by generating random data:
    
    :math:`z_1=Az_0+w,  z_0 \\sim {\\mathcal N}(r_0, \\tau_0 I), 
       w \\sim {\\mathcal N}(0, \\tau_w I)`
       
    :math:`r_1 = z_1 + {\\mathcal N}(0,\\tau_1 I)
       
    Then the method estimates :math:`z_0,z_1` from :math:`r_1,r_0` 
    and compares the expected and measured errors.
    
    :param nz0:  number of rows of :math:`z_0`
    :param nz1:  number of rows of :math:`z_1`
    :param ns:  number of columns of :math:`z_0` and :math:`z_1`
    :param Boolean map_est:  perform MAP estimation 
    :param Boolean verbose:  print results
    :param tol:  error tolerance above which test is considered to fail.    
    """            

    # Other parameters
    is_complex = False
    
    # Generate random variances
    rvar0 = 10**(np.random.uniform(-1,1,1))[0]
    rvar1 = 10**(np.random.uniform(-1,1,1))[0]
    wvar = 10**(np.random.uniform(-1,1,1))[0]
    
    # Get shapes
    if (ns == 1):
        zshape0 = (nz0,)       
        zshape1 = (nz1,)
    else:
        zshape0 = (nz0,ns)       
        zshape1 = (nz1,ns)
    Ashape = (nz1,nz0)
    
    # Generate random matrix and offset
    A = np.random.normal(0,1,Ashape)/np.sqrt(nz0)
    b = np.zeros(nz1) 
    if ns > 1:
        b = b[:,None]
    
    # Add noise on input and output
    r0 = np.random.normal(0,1,zshape0) 
    z0 = r0 + np.random.normal(0,np.sqrt(rvar0),zshape0)
    z1 = A.dot(z0) + b + np.random.normal(0,np.sqrt(wvar),zshape1)
    r1 = z1 + np.random.normal(0,np.sqrt(rvar1),zshape1)
        
    # Create linear estimator class
    Aop = trans.MatrixLT(A,zshape0)
    est = LinEstimTwo(Aop,b,wvar=wvar,map_est=map_est,z1rep_axes='all',\
                      z0rep_axes='all')
    
    # Pack the variables
    r = [r0,r1]
    rvar = [rvar0,rvar1]
    
    # Find the true solution
    # H = ||z1-A*z0-b||^2/wvar + \sum_{i=0,1} ||z-ri||^2/rvari
    H = np.zeros((nz0+nz1,nz0+nz1))
    H[:nz0,:nz0] = A.conj().T.dot(A)/wvar + np.eye(nz0)/rvar0
    H[:nz0,nz0:] = -A.conj().T/wvar 
    H[nz0:,:nz0] = -A/wvar 
    H[nz0:,nz0:] = np.eye(nz1)*(1/wvar + 1/rvar1) 
    if ns > 1:
        g = np.zeros((nz0+nz1,ns))
        g[:nz0,:] = -A.conj().T.dot(b)/wvar + r0/rvar0
        g[nz0:,:] = b/wvar + r1/rvar1 
    else:
        g = np.zeros(nz0+nz1)
        g[:nz0] = -A.conj().T.dot(b)/wvar + r0/rvar0
        g[nz0:] = b/wvar + r1/rvar1 
            
    zhat_true = np.linalg.solve(H,g)
    if ns > 1:
        zhat0_true = zhat_true[:nz0,:]
        zhat1_true = zhat_true[nz0:,:]
    else:
        zhat0_true = zhat_true[:nz0]
        zhat1_true = zhat_true[nz0:]        
    
    zcov = np.diag(np.linalg.inv(H))
    zhatvar0_true = np.mean(zcov[:nz0])
    zhatvar1_true = np.mean(zcov[nz0:])
    
    # Compute the cost of the first order terms
    cost_out = np.linalg.norm(zhat1_true-A.dot(zhat0_true)-b)**2/wvar
    cost0 = np.linalg.norm(zhat0_true-r0)**2/rvar0
    cost1 = np.linalg.norm(zhat1_true-r1)**2/rvar1
    cost_true = cost_out+cost0+cost1
    
    # Compute the cost of the second order terms
    if is_complex:
        cscale = 1
    else:
        cscale = 2
    cost_true += nz1*ns*np.log(cscale*np.pi*wvar)
    if not map_est:
        lam = np.linalg.eigvalsh(H)
        cost_true -= ns*np.sum(np.log(cscale*np.pi/lam))
    
    cost_true /= cscale
        
    zhat, zhatvar, cost = est.est(r,rvar,return_cost=True)
    zhat0, zhat1 = zhat
    zhatvar0,zhatvar1 = zhatvar
    
    zerr0 = np.linalg.norm(zhat0-zhat0_true)    
    zerr1 = np.linalg.norm(zhat1-zhat1_true)    
    if verbose:
        print("zhat error:    {0:12.4e}, {0:12.4e}".format(zerr0,zerr1))
    if (zerr0 > tol) or (zerr1 > tol):
        raise common.TestException("Error in first order terms")
    
    zerr0 = np.abs(zhatvar0-zhatvar0_true)
    zerr1 = np.abs(zhatvar1-zhatvar1_true)
    if verbose:    
        print("zhatvar error: {0:12.4e}, {0:12.4e}".format(zerr0,zerr1))
    if (zerr0 > tol) or (zerr1 > tol):
        raise common.TestException("Error in second order terms")
    
    cost_err = np.abs(cost-cost_true)
    if verbose:
        print("cost error:    {0:12.4e}".format(cost_err))
    if (zerr0 > tol) or (zerr1 > tol):
        raise common.TestException("Error in cost evaluation")

def lin_two_test_mult(verbose=False):
    """
    Unit tests for the linear estimator class
    
    This calls :func:`lin_two_test` with multiple different paramter values
    """
    lin_two_test(nz0=100,nz1=200,ns=10,map_est=True,verbose=verbose)
    lin_two_test(nz0=200,nz1=100,ns=10,map_est=True,verbose=verbose)
    lin_two_test(nz0=100,nz1=200,ns=1,map_est=True,verbose=verbose)
    lin_two_test(nz0=200,nz1=100,ns=1,map_est=True,verbose=verbose)
    lin_two_test(nz0=100,nz1=200,ns=10,map_est=False,verbose=verbose)
    lin_two_test(nz0=200,nz1=100,ns=10,map_est=False,verbose=verbose)

