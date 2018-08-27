"""
test_lin_two.py:  Test suite for the linear estimator :class:`LinEstimTwo`
"""
from __future__ import print_function

import unittest
import numpy as np

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp
        
def lin_two_test(nz0=100,nz1=200,ns=10,map_est=False,verbose=False,\
    tol1=1e-3,tol2=1e-3,tolc=1e-3,est_meth='svd',nit_cg=100,\
    wvar=None):
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
    :param tol1:  relative error on the first-order terms for test failure
    :param tol2:  relative error on the second-order terms for test failure
       This is approximately computed for the CG method, so this tolerance
       should be set high
    :param tolc:  relative error on the costfor test failure
    :param est_meth:  estimation method.  Either `svd` or `cg` corresponding 
       to whether the method is SVD or conjugate-gradient
    :param nit_cg:  number of CG iterations
    :param wvar:  Noise variance.  Set to `None` to randomly select
    """            

    # Other parameters
    is_complex = False
    
    # Generate random variances
    rvar0 = 10**(np.random.uniform(-1,1,1))[0]
    rvar1 = 10**(np.random.uniform(-1,1,1))[0]
    if wvar is None:
        wvar = 10**(np.random.uniform(-1,1,1))[0]
    wvar_zero = (wvar <= 0)
        
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
    if wvar_zero:
        z1 = A.dot(z0) + b
    else:
        z1 = A.dot(z0) + b + np.random.normal(0,np.sqrt(wvar),zshape1)        
    r1 = z1 + np.random.normal(0,np.sqrt(rvar1),zshape1)
        
    # Create linear estimator class
    var_axes = ['all', 'all']
    Aop = vp.trans.MatrixLT(A,zshape0)
    est = vp.estim.LinEstTwo(Aop,b,wvar=wvar,map_est=map_est,var_axes=var_axes,\
                      est_meth=est_meth,nit_cg=nit_cg,atol_cg=1e-9)
    
    # Pack the variables
    r = [r0,r1]
    rvar = [rvar0,rvar1]
    
    if wvar_zero:
        # Find the true solution for the case wvar = 0
        # J = ||A*z0+b-r1||^2/rvar1 + ||z0-r0||^2/rvar0
        H = A.conj().T.dot(A)/rvar1 + np.eye(nz0)/rvar0
        g = A.conj().T.dot(r1-b)/rvar1 + r0/rvar0
        
        zhat0_true = np.linalg.solve(H,g)
        zhat1_true = A.dot(zhat0_true)
        
        Q = np.linalg.inv(H)
        zhatvar0_true = np.mean(np.diag(Q))
        zhatvar1_true = np.mean(np.diag(A.dot(Q.dot(A.conj().T))))
        
        cost0 = np.linalg.norm(zhat0_true-r0)**2/rvar0
        cost1 = np.linalg.norm(zhat1_true-r1)**2/rvar1
        cost_true = cost0+cost1
        if not is_complex:
            cost_true = 0.5*cost_true
            
    else:
    
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
    
    zerr0 = np.linalg.norm(zhat0-zhat0_true)**2/np.linalg.norm(zhat0_true)**2
    zerr1 = np.linalg.norm(zhat1-zhat1_true)**2/np.linalg.norm(zhat1_true)**2 
    if verbose:
        print("zhat error:    {0:12.4e}, {1:12.4e}".format(zerr0,zerr1))
    if (zerr0 > tol1) or (zerr1 > tol1):
        raise vp.common.TestException("Error in first order terms")
    
    zerr0 = np.sum(np.abs(zhatvar0-zhatvar0_true))/np.sum(zhatvar0_true)
    zerr1 = np.sum(np.abs(zhatvar1-zhatvar1_true))/np.sum(zhatvar1_true)
    fail = (zerr0 > tol2) or (zerr1 > tol2)
    if verbose or fail:
        print("zhatvar error: {0:12.4e}, {1:12.4e}".format(zerr0,zerr1))
    if fail:
        raise vp.common.TestException("Error in second order terms")
    
    cost_err = np.abs(cost-cost_true)/np.abs(cost_true)
    fail = (cost_err > tolc)
    if verbose or fail:
        print("cost error:    {0:12.4e}".format(cost_err))
    if fail:
        raise vp.common.TestException("Error in cost evaluation")

class TestCases(unittest.TestCase):
    def test_lin_estim_two_svd(self):
        """
        Calls the lin_two_test with different parameter values using the 
        SVD method
        """
        verbose = False
        lin_two_test(nz0=100,nz1=200,ns=10,map_est=True,verbose=verbose)
        lin_two_test(nz0=200,nz1=100,ns=10,map_est=True,verbose=verbose)
        lin_two_test(nz0=100,nz1=200,ns=1,map_est=True,verbose=verbose)
        lin_two_test(nz0=200,nz1=100,ns=1,map_est=True,verbose=verbose)
        lin_two_test(nz0=100,nz1=200,ns=10,map_est=False,verbose=verbose)
        lin_two_test(nz0=200,nz1=100,ns=10,map_est=False,verbose=verbose)
    
    def test_lin_estim_two_cg(self):
        verbose = False
        lin_two_test(nz0=100,nz1=200,ns=10,map_est=True,verbose=verbose,\
            est_meth='cg',nit_cg=100,tol1=1e-6,tol2=0.1,tolc=1e-6)
        lin_two_test(nz0=100,nz1=200,ns=10,map_est=True,verbose=verbose,\
            est_meth='cg',nit_cg=100,tol1=1e-6,tol2=0.1,tolc=1e-6,wvar=0)
            
        
if __name__ == '__main__':    
    unittest.main()
    
