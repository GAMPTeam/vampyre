"""
mlp_test.py:  Tests the ML-VAMP algorithm for a synthetic MLP model

This program is used to generate the data for the paper "Inference 
in Deep networks in High Dimensions".  

The paper generates a random MLP, runs the ML-VAMP algorithm and compares 
the measured and predicted MSE.    The results are saved in the pickle file,
randmlp_sim.pkl.  This file is then used in mlp_plot.py
"""

# Import packages

# Add the vampyre path to the system path
import os
import sys
vp_path = os.path.abspath('../../')
if not vp_path in sys.path:
    sys.path.append(vp_path)
import vampyre as vp

# Load the other packages
import numpy as np
import matplotlib.pyplot as plt

import pickle
import randmlp
import argparse

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Simulate ML-VAMP on a random MLP.')
parser.add_argument('-nit',action='store',default=50,type=int,\
    help='number of MLVAMP iters per trial')
parser.add_argument('-ntrial',action='store',default=20,type=int,\
    help='number of trials per measurement')
parser.add_argument('-ny',action='store',nargs='+',\
    default=[10,30,50,100,200,300],type=int,\
    help='list of values of ny')
parser.add_argument('-damp',action='store',default=0.7,type=float,\
    help='damping on the first-order terms for small networks')
parser.add_argument('-damp_var',action='store',default=0.5,type=float,\
    help='damping on the second-order terms for small networks')
parser.add_argument('-save_dat', dest='save_dat', action='store_true',\
    help="Save results in a pickle file")
parser.set_defaults(save_dat=False)    
    
        
args = parser.parse_args()
nit = args.nit
ntrial = args.ntrial
damp = args.damp
damp_var = args.damp_var
save_dat = args.save_dat
ny_test = np.array(args.ny)
ntest = len(ny_test)

# Set random seed for reproducibility
#np.random.seed(0)

class MLPSim:

    def __init__(self,nit=50,rvarmin=0,damp=0.95,damp_var=0.95,ny=300):
        """
        Class for running one MLP simulation
        
        :param nit:  Number of MLVAMP iterations
        :param damp:  Damping on the first-order terms 
        :param damp_var:  Damping on the second-order terms
        :param rvarmin:  Minimum variance.
        :param ny:  Number of output measurements
        """
        
        # Dimensions
        self.nin  = 20           # dimension of the latent variables, dim(z0)
        self.nhid = [100,500]    # dimension of the hidden units in the subsequent layers, dim(z_1)=dim(z_2)=100, dim(z_3)=dim(z_4)=500,
        self.nx = 784            # dimension of the unknown signal, dim(x)=dim(z_5)
        self.ny = ny             # number of measurements = dimension(y)
        
        # Save configurable parameters
        self.nit = nit
        self.rvarmin = rvarmin
        self.damp = damp
        self.damp_var = damp_var
        
        # Other parameters
        self.ns = 10             # Number of samples to generate
        self.snr = 20            # SNR in dB
        
        # Sparsity target levels
        self.sparse_tgt = [0.4,0.4]
        
        
    def debias_mse(self,zhat,ztrue):
        tol = 1e-8
        zcorr = np.abs(np.mean(zhat.conj()*ztrue,axis=0))**2
        zpow = np.mean(ztrue.conj()*ztrue,axis=0)
        zhatpow = np.mean(zhat.conj()*zhat,axis=0)
        if np.any(zpow < tol) or np.any(zhatpow < tol):
            dmse = 0
        else:
            rho = 1-zcorr/zpow/zhatpow
            dmse = 10*np.log10(np.mean(rho))
        return dmse
        
        
    def sim(self):

        """
        Generate random network
        """                
        # Generate a random network
        mlp = randmlp.RandomMLP(self.nin,self.nhid,self.nx,sparse_tgt=self.sparse_tgt)
        mlp.gen_weigths()
        
        # Extract the weights and biases
        #Ws = mlp.Ws
        bs = mlp.bs
        
        # Generate random samples
        mlp.ns = self.ns
        zs = mlp.run()

        # Get the unknown vector x from the final layer of the MLP
        x = zs[-1]
        
        # Get bias in final layer
        nlayer = mlp.nlayers
        bout = bs[nlayer]
        
        # Generate a random sensing matrix 
        A = np.random.normal(0,1,(self.ny,self.nx))/np.sqrt(self.nx)
        
        # Compute the noise variance.  For the noise variance we remove the bias
        y0 = A.dot(x-bout[:,None])
        wvar = 10**(-0.1*self.snr)*np.mean(y0**2)
        
        # Add noise
        w = np.random.normal(0,np.sqrt(wvar),(self.ny,self.ns))
        y = A.dot(x) + w
        
        # Create estimator list
        est_list = []
        
        # Initial estimator
        est0 = vp.estim.GaussEst(0,1,zs[0].shape)
        est_list.append(est0)          
            
        for i in range(mlp.nlayers):
            
            # Get shape
            zshape0 = zs[2*i].shape
            zshape1 = zs[2*i+1].shape
            
            # Add linear layer
            Wi = mlp.Ws[i]
            bi = mlp.bs[i]
            Wiop = vp.trans.MatrixLT(Wi,zshape0)
            esti = vp.estim.LinEstimTwo(Wiop,bi[:,None])
            est_list.append(esti)
            
            # Add the ReLU layer
            esti = vp.estim.ReLUEstim(zshape1,map_est=False)
            est_list.append(esti)
                
        # Add the final linear layer with measurement
        i = mlp.nlayers
        zshape0 = zs[2*i].shape
        Wi = mlp.Ws[i]
        bi = mlp.bs[i]
        Wiop = vp.trans.MatrixLT(A.dot(Wi),zshape0)
        esti = vp.estim.LinEstim(Wiop,y-A.dot(bi[:,None]),wvar)
        est_list.append(esti)
        
        # Create the msg handlers
        nvar = 2*mlp.nlayers+1
        msg_hdl_list = []
        for i in range(nvar):
            msg_hdl = vp.estim.MsgHdlSimp(shape=zs[i].shape,\
                damp=self.damp,damp_var=self.damp_var,rvar_min=self.rvarmin)
            msg_hdl_list.append(msg_hdl)
            
        
        # Create the MLVamp solver
        #self.nit=3
        solver = vp.solver.MLVamp(est_list,msg_hdl_list,comp_cost=True,\
                hist_list=['zhat','rvarfwd','rvarrev','zhatvar'],nit=self.nit)
        self.solver = solver
                
        # Run the solver
        solver.solve()          
        
        # Extract the estimate
        zhatvar = solver.hist_dict['zhatvar']
        zhat = solver.hist_dict['zhat']
    
    
        # Compute the powers
        zpow = np.zeros(nvar)
        for i in range(nvar):
            zpow[i] = np.mean(zs[i]**2)
            
        nit2 = len(zhat)
        self.mse_act = np.zeros((nit2,nvar))
        self.mse_pred = np.zeros((nit2,nvar))
        for it in range(nit2):
            for i in range(nvar):
                zhati = zhat[it][i]
                #zerr = np.mean(np.abs(zhati - zs[i])**2)
                self.mse_act[it,i] = self.debias_mse(zhati,zs[i])        
                self.mse_pred[it,i] = 10*np.log10(np.mean(zhatvar[it][i])/zpow[i])      


# Create the MLP simulation object
mlpsim = MLPSim(nit=nit)

# Get the dimensions
nvar = 2*len(mlpsim.nhid)+1
nit2 = 2*mlpsim.nit

# Set up matrices
mse_act = np.zeros((nit2,nvar,ntrial,ntest))
mse_pred = np.zeros((nit2,nvar,ntrial,ntest))

for itest in range(ntest):
    
    # Set damping.  We apply the additional damping only for small networks
    ny = ny_test[itest]
    if (ny < 100):
        damp0 = damp
        damp_var0 = damp_var
    else:
        damp0 = 0.95
        damp_var0 = 0.95
    
    # Create the MLP simulation object
    mlpsim = MLPSim(nit=nit,rvarmin=0.001,damp=damp0,damp_var=damp_var0,ny=ny)    
  
    for it in range(ntrial):
        
        # Run it
        mlpsim.sim()
        
        # Get the MSE values
        mse_acti = mlpsim.mse_act
        mse_predi = mlpsim.mse_pred
        
        # Check if nan
        if np.any(np.isnan(mse_acti)):
            mse_acti = np.zeros((nit2,nvar))
            mse_predi = np.zeros((nit2,nvar))
        
        # Save the results
        mse_act[:,:,it,itest] = mse_acti
        mse_pred[:,:,it,itest] = mse_predi
    
        # Get the final results    
        mse_actf = mse_acti[-1,:]
        mse_predf = mse_predi[-1,:]
        ivar = 0
        print("ny={0:3d}, it={1:2d}, mse act={2:7.2f}  pred={3:7.2f}".format(\
            ny, it, mse_actf[ivar],mse_predf[ivar]))
            
# Compute the median MSE over the trials
mse_act_med = np.median(mse_act,axis=2)    
mse_pred_med = np.median(mse_pred,axis=2)    

# Plot MSE vs. iteration
if 0:    
    t = np.array(range(nit2))
    plt.figure(figsize=(10,5))
    ivar = 0
    itest = 2
    plt.plot(t, mse_act_med[:,ivar,itest],'-o')
    plt.plot(t, mse_pred_med[:,ivar,itest],'-s')
    plt.legend(['actual', 'SE'])
    plt.grid()
    plt.xlabel('Half iteration')
    plt.ylabel('Normalized MSE (dB)')    

# Compute median over iteration
mse_act_med1 = mse_act_med[-1,:,:]
mse_pred_med1 = mse_pred_med[-1,:,:]
print("Summary")
for i in range(ntest):
    print("ny={0:3d} mse act={1:7.2f} pred={2:7.2f} err={3:7.2f}".format(\
        ny_test[i], mse_act_med1[ivar,i], mse_pred_med1[ivar,i], \
        mse_act_med1[ivar,i]-mse_pred_med1[ivar,i]))

# Plot final MSE vs. number of measurements
if 0:
    plt.plot(ny_test, mse_act_med1[ivar,:],'o', linewidth=2)
    plt.plot(ny_test, mse_pred_med1[ivar,:],'-s', linewidth=2)
    plt.legend(['actual', 'SE'])
    plt.grid()
    plt.xlabel('Num measurements')
    plt.ylabel('Normalized MSE (dB)')    
    

# Save results
if save_dat:
    fp = open('randmlp_sim.pkl', 'wb')
    pickle.dump([ny_test,mse_act,mse_pred], fp)
    fp.close()
    print("Results saved in randmlp_sim.pkl")
else:
    print("Results not saved.  Used [-save_dat] option.")
