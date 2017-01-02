"""
vamp_sparse_test.py:  tests for VAMP for sparse recovery
"""

import vampyre as vp
import numpy as np
                   
def sparse_inv(nz=1000,ny=500,ns=10, snr=30, verbose=False, mse_tol=-29):    
    """
    Test VAMP on a sparse inverse problem
    
    In this test, the input :math:`z` is a Bernoulli-Gaussian and 
    :math:`y=Az+w` where :math:`w` is Gaussian noise and :math:`A` is an
    i.i.d. Gaussian matrix
    
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
    zshape = (nz,ns)
    yshape = (ny,ns)
    Ashape = (ny,nz)

    # Input parameters
    sparse_rat = 0.1   # sparsity ratio
    zmean1 = 0         # mean for the active components
    zvar1 = 1          # variance for the active components
   
    # Generate sparse input data
    z1 = np.random.normal(zmean1, np.sqrt(zvar1), zshape)
    u = np.random.uniform(0, 1, zshape) < sparse_rat
    z = z1*u
    
    # Create a random transform
    A = np.random.normal(0,np.sqrt(1/nz), Ashape)
    
    # Create output
    y0 = A.dot(z) 
    wvar = np.power(10,-0.1*snr)*np.mean(np.abs(y0)**2)
    y = y0 + np.random.normal(0,np.sqrt(wvar),yshape)

    # Create the distribution for the two components of the input
    est0 = vp.estim.DiscreteEst(0,1,zshape)
    est1 = vp.estim.GaussEst(zmean1,zvar1,zshape)
        
    # Create the mixture distribution 
    est_list = [est0, est1]
    pz = np.array([1-sparse_rat, sparse_rat])
    est_in = vp.estim.MixEst(est_list, w=pz)
    
    # Create output estimator
    Aop = vp.trans.MatrixLT(A,zshape)
    est_out = vp.estim.LinEstim(Aop,y,wvar,map_est=map_est)

    # Create the variance handler
    msg_hdl = vp.estim.MsgHdlSimp(map_est=map_est, is_complex=is_complex,\
                                  shape=zshape,damp=0.95)

    # Create and run the solver
    solver = vp.solver.Vamp(est_in,est_out,hist_list=['zhat'],\
             comp_cost=True, msg_hdl=msg_hdl)
    solver.solve()
    
    # Compute the MSE as a function of the iteration
    zhat_hist = solver.hist_dict['zhat']
    nit = len(zhat_hist)
    zpow = np.mean(np.abs(z)**2)
    mse = np.zeros(nit)
    for it in range(nit):
        zerr = np.mean(np.abs(zhat_hist[it]-z)**2)
        mse[it] = 10*np.log10(zerr/zpow)
    
    if verbose or (mse[-1] > mse_tol):
        print("Final MSE = {0:f}".format(mse[-1]))        
    
    # Check final error if test passed
    if mse[-1] > mse_tol:
        raise vp.common.TestException("MSE exceeded expected value")
        


