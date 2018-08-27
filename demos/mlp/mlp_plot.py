"""
mlp_plot.py:  Creates plots for the results of the random MLP simulation.

You must first run the program mlp_test.py to generate the results pickle
file, randmlp_sim.pkl.
"""

# Load the packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib 
import argparse

parser = argparse.ArgumentParser(description='Plots the results of mlp_test.py')
parser.add_argument('-plt_iter', dest='plt_iter', action='store_true',\
    help="Plots MSE vs. iteration")
parser.set_defaults(plt_iter=False)

parser.add_argument('-plt_meas', dest='plt_meas', action='store_true',\
    help="Plots MSE vs. number of measurements")
parser.set_defaults(plt_meas=False)

parser.add_argument('-itest',action='store',default=2,type=int,\
    help='Index of test for the MSE vs. iteration plot')
parser.add_argument('-ivar',action='store',default=0,type=int,\
    help='Index of variable to plot')

args = parser.parse_args()

# Set fontsize
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams.update({'font.size': 20})

# Load the saved data
fp = open('randmlp_sim.pkl', 'rb')
[ny_test,mse_act,mse_pred] = pickle.load(fp)
fp.close()

# Compute the median
mse_act_med = np.median(mse_act,axis=2)    
mse_pred_med = np.median(mse_pred,axis=2)  

"""
Plots the MSE vs. iteration
"""
if args.plt_iter:
    nit2 = mse_act_med.shape[0]
    t = np.array(range(nit2))
    plt.figure(figsize=(10,5))
    ivar = args.ivar
    itest = args.itest
    plt.plot(t, mse_act_med[:,ivar,itest],'-o',linewidth=3)
    plt.plot(t, mse_pred_med[:,ivar,itest],'-s',linewidth=3)
    plt.legend(['actual', 'pred'])
    plt.grid()
    plt.xlabel('Half iteration')
    plt.ylabel('Normalized MSE (dB)')
        
    fn = "randmlp_iter_ny{0:d}_ivar{1:d}.png".format(ny_test[itest],ivar)
    plt.savefig(fn, bbox_inches='tight')    
    print("Created file: {0:s}".format(fn))
    
"""
Plots MSE vs number of measurements
"""
if args.plt_meas:
    # Compute median over iteration
    mse_act_med1 = mse_act_med[-1,:,:]
    mse_pred_med1 = mse_pred_med[-1,:,:]
    
    ntest = mse_act_med1.shape[1]
    plt.figure(figsize=(10,5))
    ivar = args.ivar
    plt.plot(ny_test, mse_act_med1[ivar,:],'-o',linewidth=3)
    plt.plot(ny_test, mse_pred_med1[ivar,:],'-s',linewidth=3)
    plt.legend(['actual', 'pred'])
    plt.grid()
    plt.xlabel('Num measurements')
    plt.ylabel('Normalized MSE (dB)')   
    #plt.show()
    
    fn = "randmlp_meas_ivar{0:d}.png".format(ivar)
    plt.savefig(fn, bbox_inches='tight')        
    print("Created file: {0:s}".format(fn))

if (not args.plt_iter) and (not args.plt_meas):
    print("No plot selected.  Add option -plt_iter and/or -plt_meas.")