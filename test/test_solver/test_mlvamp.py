import numpy as np
import vampyre.common as common
import vampyre.trans as trans
from vampyre.solver import MLVamp
import vampyre.estim as estim

def test_mlvamp_probit(nz0=512,nz1=4096,ns=1,snr=40,verbose=False,\
    mse_tol=[-20,-20]):
    """
    Test for the MLVAMP method using a simple Probit GLM
    
    In this test, :nath:`z_0` is a sparse Bernoulli-Gaussian matrix,
    :math:`z_1 = Az_0 + w`, where :math:`A` is a random iid Gaussian
    matrix, :math:`w` is Gaussian noise and :math:`y= (z_1 > 0)`.
    The MLVAMP solver, :class:`MLVamp` is used to estimate :math:`z_0`
    and :math:`z_1` from :math:`y`.  The test passes if the MSE
    is below the required tolerances.
    
    :param nz0:  number of elements for each column of :math:`z_0`
    :param nz1:  number of elements for each column of :math:`z_1`
    :param ns:  number of columns of :math:`z_0` and :math:`z_1`
    :param snr:  SNR in dB
    :param mse_tol:  maximum MSE under which the test passes
    :param verbose:  print results
    """
    
    ny = nz1
    map_est = False
    sparse_rat = 0.1

    # Compute the dimensions
    if (ns==1):
        zshape0 = (nz0,)
        zshape1 = (nz1,)
        yshape = (ny,)
    else:
        zshape0 = (nz0,ns)
        zshape1 = (nz1,ns)
        yshape = (ny,ns)
    Ashape = (nz1,nz0)

    # Generate random input z
    #np.random.seed(40)
    zpowtgt = 2
    zmean0 = 0
    zvar0 = zpowtgt/sparse_rat
    z0 = np.random.normal(zmean0,np.sqrt(zvar0),zshape0)
    u = np.random.uniform(0,1,zshape0) < sparse_rat
    z0 = z0*u
    
    # Normalize z to the average value.  This appears necessary to get
    # very low errors.
    zpow = np.mean(z0**2,axis=0)
    if (ns > 1):
        zpow = zpow[None,:]
    z0 = z0*np.sqrt(zpowtgt/zpow)
    
    # Create a random transform and bias
    A = np.random.normal(0,np.sqrt(1/nz0), Ashape)
    b = np.zeros(zshape1)
            
    # Lienar transform
    Az0 = A.dot(z0) + b
    wvar = np.power(10,-0.1*snr)*np.mean(np.abs(Az0)**2)
    z1 = Az0 + np.random.normal(0,np.sqrt(wvar),yshape)
    
    # Save the true values
    ztrue = [z0,z1]
    
    # Signed output
    thresh = 0
    y = (z1 > thresh)
    
    # Create estimators for the input and output of the transform
    est0_gauss = estim.GaussEst(zmean0,zvar0,zshape0,map_est=map_est)
    est0_dis = estim.DiscreteEst(0,1,zshape0)
    est_in = estim.MixEst([est0_gauss,est0_dis],[sparse_rat,1-sparse_rat])
    
    est_out = estim.HardThreshEst(y,yshape,thresh=thresh)
    
    # Estimtor for the linear transform
    Aop = trans.MatrixLT(A,zshape0)
    est_lin = estim.LinEstimTwo(Aop,b,wvar)
    
    est_list = [est_in,est_lin,est_out]
    
    # Create the message handlers
    damp=0.95
    msg_hdl0 = estim.MsgHdlSimp(map_est=map_est, shape=zshape0,damp=damp) 
    msg_hdl1 = estim.MsgHdlSimp(map_est=map_est, shape=zshape1,damp=damp) 
    msg_hdl_list  = [msg_hdl0,msg_hdl1]
    
    solver = MLVamp(est_list,msg_hdl_list,comp_cost=True,\
        hist_list=['zhat','zhatvar','cost'])
    
    # Run the solver
    solver.solve()
    
    # Compute the predicted and true MSE
    zhat = solver.hist_dict['zhat']
    zhatvar = np.array(solver.hist_dict['zhatvar'])
    nvar = len(ztrue)
    nit2 = len(zhat)
    mse_pred = np.zeros((nit2,nvar))
    mse_act = np.zeros((nit2,nvar))
    for i in range(nvar):
        zpow = np.mean(np.abs(ztrue[i])**2)
        mse_pred[:,i] = 10*np.log10(zhatvar[:,i]/zpow)
        for it in range(nit2):        
            zi = zhat[it][i]
            zerr = np.mean(np.abs(zi-ztrue[i])**2)
            mse_act[it,i] = 10*np.log10(zerr/zpow)
            
    # Test if fails and print results
    for i in range(nvar):        
        fail = (mse_act[-1,i] > mse_tol[i])
        if fail or verbose:
            print("MSE {0:d} actual: {1:7.2f} pred: {1:7.2f}".format(\
                i, mse_act[-1,i], mse_pred[-1,i]) )
        if fail:
            common.TestException("MSE exceeded tolerance")
                

