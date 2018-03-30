from __future__ import division
"""
test_vamp_probit.py:  tests for ML-VAMP for probit estimation
"""

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp

# Add other packages
import numpy as np
import unittest

def debias_mse(zhat,ztrue):
    """
    If zhat and ztrue are 1D vectors, the function computes the *debiased normalized MSE* defined as:
    
        dmse_lin = min_c ||ztrue-c*zhat||^2/||ztrue||^2 = (1-|zhat'*ztrue|^2/||ztrue||^2||zhat||^2)
        
    The function returns the value in dB:  dmse = 10*log10(dmse_lin)
    
    If zhat and ztrue are matrices, dmse_lin is computed for each column and then averaged over the columns
    """
    zcorr = np.abs(np.sum(zhat.conj()*ztrue,axis=0))**2
    zhatpow = np.sum(np.abs(zhat)**2,axis=0)
    zpow = np.sum(np.abs(ztrue)**2,axis=0)
    tol = 1e-8
    if np.any(zhatpow < tol) or np.any(zpow < tol):
        dmse = 0
    else:
        dmse = 10*np.log10(np.mean(1 - zcorr/zhatpow/zpow))
    return dmse    
    
def probit_test(nz0=512,nz1=4096,ncol=10, snr=30, verbose=False, plot=False,\
    est_meth='cg', nit_cg=10, mse_tol=-20):    
    """
    Test VAMP on a sparse probit estimation problem
    
    In this test, the input :math:`z_0` is a Bernoulli-Gaussian and 
    :math:`z_1=Az_0+w` where :math:`w` is Gaussian noise and :math:`A` is an
    i.i.d. Gaussian matrix.  The problem is to estimate :math:`z_0` from
    binary measurements :math:`y=sign(z_1)`.  This is equivalent to sparse
    probit estimation in statistics.
    
    :param nz0:  number of rows of :math:`z_0`
    :param nz1:  number of rows of :math:`z_1`
    :param ncol:  number of columns of :math:`z_1` and :math:`z_0`
    :param snr: SNR in dB
    :param Boolean verbose: Flag indicating if the test results are
       to be printed.
    :param Boolean plot: Flag indicating if the test results are
       to be plot     
    :param est_meth:  Estimation method. Either `svd` or `cg`       
    :param nit_cg:  number of CG iterations    
    :param mse_tol:  MSE must be below this value for test to pass.  
    """    
 
    # Parameters
    map_est = False
    sparse_rat = 0.1
    
    # Compute the dimensions
    ny = nz1
    if (ncol==1):
        zshape0 = (nz0,)
        zshape1 = (nz1,)
        yshape = (ny,)
    else:
        zshape0 = (nz0,ncol)
        zshape1 = (nz1,ncol)
        yshape = (ny,ncol)
    Ashape = (nz1,nz0)
    
    # Generate random input z
    #np.random.seed(40)
    zpowtgt = 2
    zmean0 = 0
    zvar0 = zpowtgt/sparse_rat
    z0 = np.random.normal(zmean0,np.sqrt(zvar0),zshape0)
    u = np.random.uniform(0,1,zshape0) < sparse_rat
    z0 = z0*u
    zpow = np.mean(z0**2,axis=0)
    
    if (ncol > 1):
        zpow = zpow[None,:]
    z0 = z0*np.sqrt(zpowtgt/zpow)
    
    # Create a random transform   
    A = np.random.normal(0,np.sqrt(1/nz0), Ashape)
    b = np.random.normal(0,1,zshape1)
            
    # Lienar transform
    Az0 = A.dot(z0) + b
    wvar = np.power(10,-0.1*snr)*np.mean(np.abs(Az0)**2)
    z1 = Az0 + np.random.normal(0,np.sqrt(wvar),yshape)
    
    # Signed output
    thresh = 0
    y = (z1 > thresh)
    
    # Create estimators for the input and output of the transform
    est0_gauss = vp.estim.GaussEst(zmean0,zvar0,zshape0,map_est=map_est)
    est0_dis = vp.estim.DiscreteEst(0,1,zshape0)
    est_in = vp.estim.MixEst([est0_gauss,est0_dis],[sparse_rat,1-sparse_rat],\
        name='Input')
    
    est_out = vp.estim.BinaryQuantEst(y,yshape,thresh=thresh, name='Output')
    
    # Estimtor for the linear transform
    Aop = vp.trans.MatrixLT(A,zshape0)
    est_lin = vp.estim.LinEstTwo(Aop,b,wvar,est_meth=est_meth,nit_cg=nit_cg,\
        name ='Linear')
    
    # List of the estimators    
    est_list = [est_in,est_lin,est_out]
    
    # Create the message handler
    damp=1
    msg_hdl0 = vp.estim.MsgHdlSimp(map_est=map_est, shape=zshape0,damp=damp) 
    msg_hdl1 = vp.estim.MsgHdlSimp(map_est=map_est, shape=zshape1,damp=damp) 
    msg_hdl_list  = [msg_hdl0,msg_hdl1]
    
    ztrue = [z0,z1]
    solver = vp.solver.mlvamp.MLVamp(est_list,msg_hdl_list,comp_cost=True,\
        hist_list=['zhat','zhatvar'])

    
    # Run the solver
    solver.solve()
    
    # Get the estimates and predicted variances
    zhat_hist    = solver.hist_dict['zhat']
    zvar_hist = solver.hist_dict['zhatvar']
    
    # Compute per iteration errors
    nvar = len(ztrue)
    nit2 = len(zhat_hist)
    mse_act = np.zeros((nit2,nvar))
    mse_pred = np.zeros((nit2,nvar))
    for ivar in range(nvar): 
        zpowi = np.mean(np.abs(ztrue[ivar])**2, axis=0)
        for it in range(nit2):
            zhati = zhat_hist[it][ivar]
            zhatvari = zvar_hist[it][ivar]
            mse_act[it,ivar] = debias_mse(zhati,ztrue[ivar])
            mse_pred[it,ivar] = 10*np.log10(np.mean(zhatvari/zpowi))
            
    # Check failure
    fail = np.any(mse_act[-1,:] > mse_tol)
            
    # Display the final MSE
    if verbose or fail:
        print("z0 mse:  act: {0:7.2f} pred: {1:7.2f}".format(\
            mse_act[-1,0],mse_pred[-1,0]))
        print("z1 mse:  act: {0:7.2f} pred: {1:7.2f}".format(\
            mse_act[-1,1],mse_pred[-1,1]))
            
    
    if plot:
        import matplotlib.pyplot as plt
        t = np.array(range(nit2))
        for ivar in range(nvar):
            plt.subplot(1,nvar,ivar+1)
            zpow = np.mean(abs(ztrue[ivar])**2)
            plt.plot(t, mse_act[:,ivar], 's-')
            plt.plot(t, mse_pred[:,ivar], 'o-')
            plt.legend(['true','pred'])
        
    if fail:
        raise vp.common.VpException("Final MSE higher than expected")

class TestCases(unittest.TestCase):
    def test_mlvamp_sparse_probit(self):
        """
        Calls the probit estimation test case
        """
        #probit_test(ncol=10,est_meth='cg')    
        probit_test(ncol=10,est_meth='svd',plot=False)
        
        
if __name__ == '__main__':    
    unittest.main()




