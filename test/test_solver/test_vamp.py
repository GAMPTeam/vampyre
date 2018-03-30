"""
test_tflintrans.py:  Test suite for the linear estimator :class:`TFLinTrans`
"""
from __future__ import print_function, division

import unittest
import numpy as np

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp

            
def vamp_gauss_test(nz=100,ny=200,ns=10, snr=30, map_est=False, verbose=False,\
    is_complex=False, tol=1e-5):
    """
    Unit test for VAMP method using a Gaussian prior.
    
    In this case, we use a Gaussian prior so that the exact costs can 
    be analytically computed and compared to the output of the VAMP method.
    
    The prior is a Gaussian :math:`z \\sim {\\mathcal N}(0,\\tau_r I)`
    and the output is a Gaussian observation 
    :math:`y = Az + w`, where :math:`w {\\mathcal N}(0,\\tau_w I)`.
    
    :param nz:  number of rows of :math:`z`
    :param ny:  number of rows of :math:`y`
    :param ns:  number of columns of :math:`y` and :math:`z`
    :param snr: SNR in dB
    :param Boolean map_est map_est: Flag indicating if the cost
       is to be computed using MAP or MMSE estimation.  
    :param Boolean verbose: Flag indicating if the test results are
       to be printed.
    :param Boolean is_complex:  Use complex data for the test.
    :param tol:  Tolerance for test to pass
    """
    
    # Check parameters
    if is_complex:
        raise vp.common.VpException(\
            "Unit test currently does not support complex case")
    
    # Compute the dimensions
    if (ns==1):
        zshape = (nz,)
        yshape = (ny,)
    else:
        zshape = (nz,ns)
        yshape = (ny,ns)
    Ashape = (ny,nz)

    # Generate random input z
    zmean0 = 0
    zvar0 = 2
    z = np.random.normal(zmean0,1/np.sqrt(zvar0),zshape)

    # Create a random transform
    A = np.random.normal(0,np.sqrt(1/nz), Ashape)
    
    # Create output
    y0 = A.dot(z) 
    wvar = np.power(10,-0.1*snr)*np.mean(np.abs(y0)**2)
    y = y0 + np.random.normal(0,np.sqrt(wvar),yshape)
    
    # Construct input estimator
    est_in = vp.estim.GaussEst(zmean0,zvar0,zshape,map_est=map_est, name='Prior')
           
    # Create output estimator
    Aop = vp.trans.MatrixLT(A,zshape)
    est_out = vp.estim.LinEst(Aop,y,wvar,map_est=map_est, name='Posterior')
    
    # Create the variance handler
    msg_hdl = vp.estim.MsgHdlSimp(map_est=map_est, is_complex=is_complex, \
        shape=zshape)
    
    # Create and run the solver
    solver = vp.solver.Vamp(est_in,est_out,msg_hdl=msg_hdl,comp_cost=True)
    solver.solve()
        
    # Comzpute Gaussian estimate
    U,s,V = np.linalg.svd(A)

    # Compute the true estimate and output of solver
    Q = np.linalg.inv(zvar0*A.conj().T.dot(A) + wvar*np.eye(nz))
    b = zvar0*A.conj().T.dot(y) + wvar*zmean0
    zhat = Q.dot(b)
    zhat_err = np.linalg.norm(zhat-solver.z2)    
    if verbose:
        print("zhat error: {0:12.4e}".format(zhat_err))
    if zhat_err > tol:
        raise vp.common.TestException("Mean does not match")
        
    # Compare the true and the output of the solver
    zvar = zvar0*wvar*np.trace(Q)/nz
    zvar_err = np.abs(zvar-np.mean(solver.zvar2))
    if verbose:
        print("zvar error: {0:12.4e}".format(zvar_err))
    if zvar_err > tol:
        raise vp.common.TestException("Variance does not match")
        
    
    # Compute the true costs for the estimators
    rvar1 = np.mean(solver.rvar1)
    rvar2 = np.mean(solver.rvar2)
    r = np.minimum(ny,nz)  # number of singular values
    d = 1/(zvar0*(s**2) + wvar)
    cost1 = np.sum(np.abs(zhat-zmean0)**2)/zvar0 +\
            np.sum(np.abs(zhat-solver.r1)**2)/rvar1
    cost2 = np.sum(np.abs(y-A.dot(zhat))**2)/wvar +\
            np.sum(np.abs(zhat-solver.r2)**2)/rvar2
        
    if  map_est:        
        cost1 += ns*nz*np.log(2*np.pi*zvar0)
        cost2 += ns*ny*np.log(2*np.pi*wvar)
    else:
        cost1 += ns*nz*np.sum(np.log((zvar0+rvar1)/rvar1))
        cost2 += - ns*r*np.mean(np.log(zvar0*d)) +\
                 (ny-r)*ns*np.log(2*np.pi*wvar) - (nz-r)*ns*np.log(2*np.pi*zvar0)
                 
    # Compute the true variance costs
    v1 = np.sum(np.abs(zhat-solver.r1)**2)/rvar1         
    v2 = np.sum(np.abs(zhat-solver.r2)**2)/rvar2
         
    if map_est:
        Hgauss = 0
    else:
        v1 += nz*ns*np.mean(solver.zvar1)/rvar1
        v2 += nz*ns*np.mean(solver.zvar2)/rvar2
        Hgauss = ns*nz*(1+np.log(2*np.pi*np.mean(solver.zvar1)))
         
    # Compute total costs            
    cost_tota = cost1 + cost2 - v1 - v2 + Hgauss
    
    # Compute the expected cost directly   
    cost_tot = np.sum(np.abs(zhat-zmean0)**2)/zvar0  +\
        np.sum(np.abs(y-A.dot(zhat))**2)/wvar +\
        ns*nz*np.log(2*np.pi*zvar0) + ns*ny*np.log(2*np.pi*wvar)
    if not map_est:
        cost_tot += ns*r*np.mean(zvar0*(s**2)*d) + r*ns*np.mean(wvar*d) + (nz-r)*ns\
            -ns*r*np.mean(np.log(2*np.pi*zvar0*wvar*d)) - ns*(nz-r)*np.log(2*np.pi*zvar0)\
            -ns*nz
    
    if not is_complex:
        cost_tot *= 0.5    
        cost_tota *= 0.5

    if verbose:    
        print("costs: Direct {0:12.4e} ".format(cost_tot)+\
             "Termwise {0:12.4e} solver: {1:12.4e}".format(cost_tota,solver.cost))                        
    if np.abs(cost_tot - cost_tota) > tol:
        raise vp.common.TestException("Direct and termwise costs do not match")
    if np.abs(cost_tot - cost_tota) > tol:
        raise vp.common.TestException("Predicted cost does not match solver output")
                    
def vamp_gmm_test(nz=100,ny=200,ns=10, snr=30, verbose=False, mse_tol=-17):    
    """
    Unit test for VAMP using a Gaussian mixture model (GMM)
    
    In this test, the input :math:`z` is a Gaussian mixture with two variances,
    and the measurements are of the form :math:`y = Az + w`, where
    :math:`w` is Gaussian noise.
    
    :param nz:  number of rows of :math:`z`
    :param ny:  number of rows of :math:`y`
    :param ns:  number of columns of :math:`y` and :math:`z`
    :param snr: SNR in dB
    :param Boolean verbose: Flag indicating if the test results are
       to be printed.
    :param mse_tol:  MSE must be below this value for test to pass.  
    """                
    # Parameters        
    map_est = False
    is_complex = False
    
    # Compute the dimensions
    if (ns==1):
        zshape = (nz,)
        yshape = (ny,)
    else:
        zshape = (nz,ns)
        yshape = (ny,ns)
    Ashape = (ny,nz)

    # GMM parameters
    zmeanc = [0, 0]   # mean of each component
    zvarc = [1,0.001]  # variance in each component
    pc = [0.1,0.9]    # probability of each component
    ncomp= len(zmeanc)
    
    # Generate GMM data
    nztot = np.prod(zshape)     
    u = np.random.choice(range(ncomp),p=pc,size=nztot)
    z = np.random.randn(nztot)
    for i in range(nztot):
        j = u[i]
        z[i] = zmeanc[j] + np.sqrt(zvarc[j])*z[i]
    z = np.reshape(z,zshape)

    # Create a random transform
    A = np.random.normal(0,np.sqrt(1/nz), Ashape)
    
    # Create output
    y0 = A.dot(z) 
    wvar = np.power(10,-0.1*snr)*np.mean(np.abs(y0)**2)
    y = y0 + np.random.normal(0,np.sqrt(wvar),yshape)

    # Create a set of estimators, one for each component of the GMM
    est_list = []
    for i in range(ncomp):
        est = vp.estim.GaussEst(zmeanc[i], zvarc[i], zshape)
        est_list.append(est)
        
    # Create the GMM estimator
    est_in =  vp.estim.MixEst(est_list, w=pc)

    # Create output estimator
    Aop = vp.trans.MatrixLT(A,zshape)
    est_out = vp.estim.LinEst(Aop,y,wvar,map_est=map_est)

    # Create the variance handler
    msg_hdl = vp.estim.MsgHdlSimp(map_est=map_est, is_complex=is_complex,\
                                  shape=zshape)

    # Create and run the solver
    solver = vp.solver.Vamp(est_in,est_out,hist_list=['z2'],\
             comp_cost=True, msg_hdl=msg_hdl)
    solver.solve()
    
    # Compute the MSE as a function of the iteration
    z2_hist = solver.hist_dict['z2']
    nit = len(z2_hist)
    zpow = np.mean(np.abs(z)**2)
    mse = np.zeros(nit)
    for it in range(nit):
        zerr = np.mean(np.abs(z2_hist[it]-z)**2)
        mse[it] = 10*np.log10(zerr/zpow)
        
    if verbose:
        print("Final MSE = {0:f}".format(mse[-1]))        
    
    # Check final error if test passed
    if mse[-1] > mse_tol:
        raise vp.common.TestException("MSE exceeded expected value")
        
def vamp_bg_test(nz=1000,ny=500,ns=1, snr=30, verbose=False, pred_tol=3.0):    
    """
    Unit test for VAMP using a Gaussian mixture model (GMM)
    
    In this test, the input :math:`z` is a Gaussian mixture with two variances,
    and the measurements are of the form :math:`y = Az + w`, where
    :math:`w` is Gaussian noise.
    
    :param nz:  number of rows of :math:`z`
    :param ny:  number of rows of :math:`y`
    :param ns:  number of columns of :math:`y` and :math:`z`
    :param snr: SNR in dB
    :param Boolean verbose: Flag indicating if the test results are
       to be printed.
    :param pred_tol:  MSE must be predicted with this tolerance to pass
    """                
    # Parameters        
    map_est = False
    is_complex = False
    
    # Compute the dimensions
    if (ns==1):
        zshape = (nz,)
        yshape = (ny,)
    else:
        zshape = (nz,ns)
        yshape = (ny,ns)
    Ashape = (ny,nz)

    # BG prior parameters
    sparse_rat = 0.1   # sparsity ratio
    zmean1 = 0         # mean for the active components
    zvar1 = 1          # variance for the active components
    
    # Generate the random input
    z1 = np.random.normal(zmean1, np.sqrt(zvar1), zshape)
    u = np.random.uniform(0, 1, zshape) < sparse_rat
    z = z1*u
        
    # Create a random transform
    A = np.random.normal(0,np.sqrt(1/nz), Ashape)
    
    # Create output
    y0 = A.dot(z) 
    wvar = np.power(10,-0.1*snr)*np.mean(np.abs(y0)**2)
    y = y0 + np.random.normal(0,np.sqrt(wvar),yshape)

    # Create the BG estimator as a mixture distribution with two components
    est0 = vp.estim.DiscreteEst(0,1,zshape)
    est1 = vp.estim.GaussEst(zmean1,zvar1,zshape)
    est_list = [est0, est1]
    pz = np.array([1-sparse_rat, sparse_rat])
    est_in = vp.estim.MixEst(est_list, w=pz)

    # Create output estimator
    Aop = vp.trans.MatrixLT(A,zshape)
    est_out = vp.estim.LinEst(Aop,y,wvar,map_est=map_est)

    # Create the variance handler
    msg_hdl = vp.estim.MsgHdlSimp(map_est=map_est, is_complex=is_complex,\
                                  shape=zshape)

    # Create and run the solver
    solver = vp.solver.Vamp(est_in,est_out,hist_list=['zhat', 'zhatvar'],\
             comp_cost=True, msg_hdl=msg_hdl)
    solver.solve()
    
    # Compute the MSE as a function of the iteration
    zhat_hist = solver.hist_dict['zhat']
    zhatvar_hist = solver.hist_dict['zhatvar']
    nit = len(zhat_hist)
    zpow = np.mean(np.abs(z)**2)
    mse_pred = np.zeros(nit)
    mse_act = np.zeros(nit)
    for it in range(nit):
        zerr = np.mean(np.abs(zhat_hist[it]-z)**2)
        mse_pred[it] = np.median(10*np.log10(zhatvar_hist[it]/zpow))
        mse_act[it] = np.median(10*np.log10(zerr/zpow))
        
    # Check if prediction matches value within tolerance
    max_err = np.max(np.abs(mse_pred-mse_act))
    fail = (max_err > pred_tol)
    if verbose or fail:
        i = np.argmax(np.abs(mse_pred-mse_act))
        print("Max err={0:5.2f} on it={1:d}.  Pred={2:5.2f} Act={3:5.2f}".format(\
            max_err, i, mse_pred[i], mse_act[i]))
            

class TestCases(unittest.TestCase):
    def test_vamp_gauss(self):
        """
        Run the vamp_gauss_test with various parameters
        """        
        verbose = False
        for map_est in [True,False]:
            vamp_gauss_test(nz=100,ny=200,ns=10,map_est=map_est,verbose=verbose)
            vamp_gauss_test(nz=200,ny=100,ns=10,map_est=map_est,verbose=verbose)
            vamp_gauss_test(nz=100,ny=200,ns=1,map_est=map_est,verbose=verbose)
            vamp_gauss_test(nz=200,ny=100,ns=1,map_est=map_est,verbose=verbose)
                
    def test_vamp_gmm(self):
        """
        Run the vamp_gmm_test
        """        
        vamp_gmm_test(nz=1000,ny=500,ns=1,verbose=False)
        
    def test_vamp_bg(self):
        """
        Run VAMP with a BG prior
        """
        vamp_bg_test(nz=1000,ny=500,ns=10,verbose=False)
        
if __name__ == '__main__':    
    #vamp_bg_test(nz=1000,ny=500,ns=10,verbose=verbose)
    unittest.main()
    
