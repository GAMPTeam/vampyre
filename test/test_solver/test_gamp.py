"""
test_gamp.py:  Unit test for the GAMP method
"""

import unittest
import numpy as np

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp

                    
def gamp_gmm_test(nz=200,ny=100,ns=10, snr=30, verbose=False, mse_tol=-17, plt_results=False):    
    """
    Unit test for GAMP using a Gaussian mixture model (GMM)
    
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
    est_in =  vp.estim.MixEst(est_list, w=pc,name='input')
    
    # Create linear transform
    Aop = vp.trans.MatrixLT(A,zshape)

    # Create the output estimator
    est_out = vp.estim.GaussEst(y,wvar,yshape,name='output')

    # Create the solver
    solver = vp.solver.Gamp(est_in,est_out,Aop,hist_list=['z0','zvar0'],step=0.95,\
                                nit=50)
    
    # Run the solver
    solver.solve()
    
    # Compute the MSE as a function of the iteration
    z0_hist = solver.hist_dict['z0']
    zvar0_hist = solver.hist_dict['zvar0']
    nit = len(z0_hist)
    zpow = np.mean(np.abs(z)**2)
    mse = np.zeros(nit)
    mse_pred = np.zeros(nit)
    for it in range(nit):
        zerr = np.mean(np.abs(z0_hist[it]-z)**2)
        mse[it] = 10*np.log10(zerr/zpow)
        mse_pred[it] = 10*np.log10(np.mean(zvar0_hist[it])/zpow)

    if (plt_results):
        import matplotlib.pyplot as plt
        t = np.arange(nit)
        plt.plot(t,mse,'-o')
        plt.plot(t,mse_pred,'-s')
        plt.legend(['Actual', 'Pred'])
        plt.grid()
        
    if verbose:
        print("Final MSE = %f" % mse[-1])        
    
    # Check final error if test passed
    if mse[-1] > mse_tol:
        raise vp.common.TestException("MSE exceeded expected value")        


class TestCases(unittest.TestCase):
              
    def test_gamp_gmm(self):
        """
        Run the vamp_gmm_test
        """        
        gamp_gmm_test(nz=1000,ny=500,ns=10,verbose=False)
                
if __name__ == '__main__':    
    #vamp_bg_test(nz=1000,ny=500,ns=10,verbose=verbose)
    unittest.main()
    
