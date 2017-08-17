from __future__ import division
"""
test_emvamp_sparse.py:  tests for VAMP for sparse recovery
"""

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp

import numpy as np
import unittest
            
def sparse_em_vamp_inv(nz0=200,nz1=100,ncol=10,snr=30, verbose=False,\
    nit=40, cond_num=100, mse_tol=-17, vamp_meth='vamp', tune=False,\
    plot_res=False):               
    """
    Test EM-VAMP and EM-ML-VAMP on a sparse inverse problem
    
    In this test, the input :math:`z` is a Bernoulli-Gaussian and 
    :math:`y=Az+w` where :math:`w` is Gaussian noise and :math:`A` is an
    i.i.d. Gaussian matrix
    
    :param nz0: number of rows of :math:`z`
    :param nz1: number of rows of :math:`y`
    :param ncol:  number of columns of :math:`y` and :math:`z`
    :param snr: SNR in dB
    :param Boolean verbose: Flag indicating if the test results are
       to be printed.
    :param mse_tol:  MSE must be below this value for test to pass.  
    :param nit:  number of iterations 
    :param cond_num:  condition number of the matrix
    :param vamp_meth: solver method: 'vamp' or 'mlvamp' for 
       VAMP or ML-VAMP
    :param tune:  flag indicating if tuning is enabled.
    :param plot_res: plots results
    """                
          
    # Parameters        
    map_est = False
    is_complex = False
    
    # Compute the dimensions
    zshape0 = (nz0,ncol)
    zshape1 = (nz1,ncol)
    
    # True clusters.
    varc_lo = 1e-4       # variance of the low variance cluster
    varc_hi = 1          # variance of the high variance cluster
    prob_hi = 0.1        # probability of the high variance cluster    
    meanc = np.array([0,0])
    probc = np.array([1-prob_hi, prob_hi])    
    varc = np.array([varc_lo, varc_hi])
    nc = len(probc)    
    
    # Generate random data following the GMM model
    zlen = np.prod(zshape0)
    ind = np.random.choice(nc,zlen,p=probc)
    u = np.random.randn(zlen)
    z0 = u*np.sqrt(varc[ind]) + meanc[ind]
    z0 = z0.reshape(zshape0)
    
    # Create a random rotationally invariant transform
    # The transform is scaled so that the elements have an average value
    # of 1/nz0
    A = vp.trans.rand_rot_invariant_mat(nz1,nz0,cond_num=cond_num)
    #A = A*np.sqrt(1/np.mean(np.abs(A)**2)/nz0)
    
    # Create output
    z1 = A.dot(z0) 
    wvar = np.power(10,-0.1*snr)*np.mean(np.abs(z1)**2)
    y = z1 + np.random.normal(0,np.sqrt(wvar),zshape1)
            

    # Create the message handlers.
    # For VAMP, we just have the message handler on the input
    # For ML-VAMP we have a list, with one on the input and one on the output.
    msg_hdl0 = vp.estim.MsgHdlSimp(map_est=map_est, is_complex=is_complex,\
                                  shape=zshape0)
    if vamp_meth == 'mlvamp':
        msg_hdl1 = vp.estim.MsgHdlSimp(map_est=map_est, is_complex=is_complex,\
                                      shape=zshape1)
        msg_hdl_list = [msg_hdl0, msg_hdl1]
        
    # When tuning is enabled, set the initial conditions
    yvar = np.mean(np.abs(y)**2)    
    meanc_init = np.array([0,0])
    prob1 = np.minimum(nz1/nz0/2,0.95)
    var1 = yvar/np.mean(np.abs(A)**2)/nz0/prob1
    probc_init = np.array([1-prob1,prob1])
    varc_init = np.array([1e-4,var1])
    mean_fix = [1,0]
    var_fix = [1,0]
        
    # Create the estimator for the linear operator
    # For the ML-VAMP case, the linear operator has zero noise
    Aop = vp.trans.MatrixLT(A,zshape0)
    b = np.zeros(zshape1)
    if vamp_meth == 'mlvamp':
        est_lin = vp.estim.LinEstimTwo(Aop,b,map_est=map_est)
    elif tune:
        est_lin = vp.estim.LinEstim(Aop,y,wvar=yvar,map_est=map_est,tune_wvar=True)
    else:
        est_lin = vp.estim.LinEstim(Aop,y,wvar=wvar,map_est=map_est,tune_wvar=False)

    # Create the input estimator, based on whether tuning is used or not        
    if tune:
        est_in = vp.estim.GMMEst(shape=zshape0,\
            zvarmin=1e-6,tune_gmm=True,probc=probc_init,meanc=meanc_init, varc=varc_init,\
            mean_fix=mean_fix, var_fix=var_fix)
    else:
        # No auto-tuning.  Set estimators with the true values
        est_in = vp.estim.GMMEst(shape=zshape0, probc=probc, meanc=meanc, varc=varc,\
                tune_gmm=False)
            

    # For the ML-VAMP case, create the output estimator
    if vamp_meth == 'mlvamp':
        if tune:
            est_out = vp.estim.GaussEst(y,zvar=yvar,shape=zshape1,zmean_axes=[],tune_zvar=True)
        else:
            est_out = vp.estim.GaussEst(y,zvar=wvar,shape=zshape1,zmean_axes=[],tune_zvar=False)            
            
        # Create the estimator list
        est_list = [est_in, est_lin, est_out]
        
    # Create the solver
    if vamp_meth == 'mlvamp':
        solver = vp.solver.MLVamp(est_list,msg_hdl_list,hist_list=['zhat','zhatvar'],\
             comp_cost=True, nit=nit)
    else:
        solver = vp.solver.Vamp(est_in,est_lin,msg_hdl0,hist_list=['zhat','zhatvar'],\
             comp_cost=True, nit=nit)            
             
    # Rnu the solver
    solver.solve()
    zhat = solver.zhat    
    
    # Compute the MSE as a function of the iteration
    zhat_hist = solver.hist_dict['zhat']
    zhatvar_hist = solver.hist_dict['zhatvar']
    nit2 = len(zhat_hist)
    zpow = np.mean(np.abs(z0)**2)
    mse_act = np.zeros(nit2)
    mse_pred = np.zeros(nit2)
    for it in range(nit2):
        if vamp_meth == 'mlvamp':
            zhati = zhat_hist[it][0]
            zhatvari = zhatvar_hist[it][0]
        else:
            zhati = zhat_hist[it]
            zhatvari = zhatvar_hist[it]
            
        zerr = np.mean(np.abs(zhati-z0)**2)
        zhatvar = np.mean(zhatvari)
        mse_act[it] = 10*np.log10(zerr/zpow)
        mse_pred[it] = 10*np.log10(zhatvar/zpow)   
        
    # Compute final MSE
    fail = (mse_act[-1] > mse_tol)
    if fail or verbose:        
        print("Final MSE {0:s}:  act {1:7.2f} pred: {2:7.2f} limit: {3:7.2f}".format(\
            vamp_meth, mse_act[-1], mse_pred[-1], mse_tol)) 
    if fail:
        raise vp.common.TestException("MSE exceeded expected value")
            
    
    if plot_res:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(7,7))
        t = np.arange(len(mse_act))        
        plt.plot(t, mse_act, 's-')        
        plt.plot(t, mse_pred, 'o-')        
        plt.grid()
        plt.legend(['actual', 'pred'])



class TestCases(unittest.TestCase):
    def test_sparse_em_vamp_inv(self):
        """
        Calls the sparse inverse test
        """
        verbose = False
        plot_res = False
        sparse_em_vamp_inv(nz0=200,nz1=100,ncol=10,vamp_meth='vamp', tune=True,\
            verbose=verbose, plot_res=plot_res)
        sparse_em_vamp_inv(nz0=200,nz1=100,ncol=10,vamp_meth='mlvamp', tune=True,\
            verbose=verbose, plot_res=plot_res)        
        
        
if __name__ == '__main__':     
    unittest.main()