"""
inpaint_test.py:  Tests various inpainting algorithms 
self.xhat_mean = (1-1/t)*self.xhat_mean + 1/t*xhati"""
from __future__ import division
from __future__ import print_function


from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle
import argparse
import os
import vae
from inpaint import MapInpaint, VAMPInpaint


"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Runs various MNIST inpainting methods')
parser.add_argument('-new_dat', dest='new_dat', action='store_true',\
    help="Extracts new set of images to test")
parser.set_defaults(new_dat=False)

parser.add_argument('-map_adam', dest='run_map', action='store_true',\
    help="Runs MAP inpainting with ADAM optimizer")
parser.set_defaults(run_map=False)

parser.add_argument('-vamp', dest='run_vamp', action='store_true',\
    help="Runs ML-VAMP inpainting")
parser.set_defaults(run_vamp=False)

parser.add_argument('-sgld', dest='run_sgld', action='store_true',\
    help="Runs SGLD inpainting")
parser.set_defaults(run_sgld=False)

parser.add_argument('-vamp_map', dest='vamp_map_est', action='store_true',\
    help="Uses MAP estimation for VAMP (default is MMSE)")
parser.set_defaults(vamp_map_est=False)

parser.add_argument('-vamp_admm', dest='vamp_admm', action='store_true',\
    help="Use ADMM solver for VAMP")
parser.set_defaults(vamp_admm=False)


parser.add_argument('-plot', dest='plot_results', action='store_true',\
    help="Plots results (assuming all methods have been done)")
parser.set_defaults(plot_results=False)    

parser.add_argument('-restore', dest='restore', action='store_true',\
    help="Continue from previous run in SGLD or MAP")
parser.set_defaults(plot_results=False)    

parser.add_argument('-lr_sgd',action='store',default=0.001,type=float,\
    help='step-size for SGLD')
parser.add_argument('-nsteps_sgld',action='store',default=10000,type=int,\
    help='total number of steps for SGLD')
parser.add_argument('-nsteps_burn',action='store',default=5000,type=int,\
    help='number of steps ignored for averaging in SGLD')
    
    
args = parser.parse_args()

# Extract parameters
new_dat = args.new_dat
run_map = args.run_map
run_sgld = args.run_sgld
run_vamp = args.run_vamp
vamp_map_est = args.vamp_map_est
plot_results = args.plot_results
vamp_admm = args.vamp_admm
lr_sgd = args.lr_sgd
nsteps_sgld = args.nsteps_sgld
nsteps_burn = args.nsteps_burn
restore = args.restore

# Data dimensions
npix = 784  # number of pixels per image
nrow = 28   # number of pixels per row
row0 = 10   # First row to erase
row1 = 20   # Last row to erase
erase_pix0 = nrow*row0
erase_pix1 = nrow*row1
Ierase = range(erase_pix0,erase_pix1)
Ikeep = np.setdiff1d(range(npix), Ierase)

# Dimensions of the layers        
enc_dim = [784,400,20]
dec_dim = [20,400,784]

"""
Load images for testing
"""

# Get true images
batch_size=100
fn = 'xtrue.p'
if new_dat:
    # Load MNIST
    if not 'mnist' in locals():
        mnist = input_data.read_data_sets('MNIST')
    batch = mnist.test.next_batch(batch_size)
    xtrue = batch[0]
    pickle.dump([xtrue], open(fn, "wb"))
else:
    if not os.path.isfile(fn):
        msg = "File {0:s} does not exist.  Use [-new_dat] option".format(fn)
        raise Exception(msg)
    print("Loading test images...")
    xtrue = pickle.load(open(fn,"rb"))[0]
    
# Generate erased images
xerase = np.copy(xtrue)
xerase[:,Ierase] = 0.5

"""
SGLD inpainting 
"""
fn = 'sgld_est.p'
if run_sgld:
    print("Running SGLD...")
    map_inpaint = MapInpaint(xtrue,erase_pix0=280, erase_pix1=560,\
        nsteps_init=500, n_steps=nsteps_sgld,lr_adam=0.01,lr_sgd=lr_sgd,\
        nsteps_burn=nsteps_burn,restore=restore)
    map_inpaint.reconstruct()
    xhat_sgld = map_inpaint.xhat
    zhat0_sgld = map_inpaint.zhat0
    zhat0_var_sgld = map_inpaint.zhat0_var
    #zsamp_hist = np.array(map_inpaint.vae_net.hist_dict['zsamp'])
    with open(fn, "wb") as fp:
        pickle.dump([xhat_sgld,zhat0_sgld,zhat0_var_sgld], fp)
elif plot_results:
    if not os.path.isfile(fn):
        msg = "SGLD has not been run. Use the [-sgld] option".format(fn)
        raise Exception(msg)           
    print("Loading SGLD results...")
    with open(fn,"rb") as fp:
        xhat_sgld,zhat0_sgld,zhat0_var_sgld = pickle.load(fp)
else:
    print("Skipping SGLD.  To run, use the [-sgld] option.")
        
"""
MAP inpainting with the ADAM optimizer
"""
fn = 'map_est.p'
if run_map:
    print("Running MAP estimation with the ADAM optimizer...")
    map_inpaint = MapInpaint(xtrue,erase_pix0=280, erase_pix1=560,\
        n_steps=500,lr_adam=0.01,recon_mode='map')
    map_inpaint.reconstruct()
    xhat_map = map_inpaint.xhat
    with open(fn, "wb") as fp:
        pickle.dump([xhat_map], fp)
elif plot_results:
    if not os.path.isfile(fn):
        msg = "MAP estimation has not been run. Use the [-map_adam] option".format(fn)
        raise Exception(msg)           
    print("Loading MAP estimation results...")
    with open(fn,"rb") as fp:
        xhat_map = pickle.load(fp)[0]
else:
    print("Skipping MAP estimation.  To run use the [-map_adam] option")
        
"""
ML-VAMP inpainting
"""
fn = 'vamp_est.p'
if run_vamp:
    print("Running ML-VAMP estimation...")
    if vamp_map_est:
        n_steps = 500
    else:
        n_steps = 100        
    vamp_inpaint = VAMPInpaint(xtrue,erase_pix0=280, erase_pix1=560,\
        n_steps=n_steps, map_est=vamp_map_est, admm=vamp_admm)
    vamp_inpaint.reconstruct()
    xhat_vamp = vamp_inpaint.xhat
    zhatvar = vamp_inpaint.zhatvar
    zhat0_var_vamp = np.mean(zhatvar[:,0,:],axis=1)    
    with open(fn, "wb") as fp:
            pickle.dump([xhat_vamp,zhatvar,zhat0_var_vamp], fp)
elif plot_results:
    if not os.path.isfile(fn):
        msg = "ML-VAMP estimation has not been run. Use the [-vamp] option".format(fn)
        raise Exception(msg)           
    print("Loading ML-VAMP estimation results...")
    with open(fn,"rb") as fp:
        xhat_vamp,zhatvar,zhat0_var_vamp = pickle.load(fp)
else:
    print("Skipping ML-VAMP estimation.  To run, use the [-vamp] option.")        
    
"""
Plot the results
"""
if plot_results:
    import matplotlib.pyplot as plt
                
    # Plot results
    nplot = 10
    nrow = 5
    fontsize = 16
    plt.figure(figsize=(7,7))
    for iplt in range(nplot):
        # True
        plt.subplot(nplot,nrow,nrow*iplt+1)
        vae.plt_digit(xtrue[iplt,:])
        if iplt == 0:
            plt.title("True",fontsize=fontsize)
        
        # Erased
        plt.subplot(nplot,nrow,nrow*iplt+2)
        vae.plt_digit(xerase[iplt,:])
        if iplt == 0:
            plt.title("Erased",fontsize=fontsize)

        
        # MAP reconstruction
        plt.subplot(nplot,nrow,nrow*iplt+3)
        vae.plt_digit(xhat_map[iplt,:])
        if iplt == 0:
            plt.title("MAP",fontsize=fontsize)
        
        # SGLD reconstruction
        plt.subplot(nplot,nrow,nrow*iplt+4)
        vae.plt_digit(xhat_sgld[iplt,:])
        if iplt == 0:
            plt.title("SGLD",fontsize=fontsize)
        
        # VAMP reconstruction
        plt.subplot(nplot,nrow,nrow*iplt+5)
        vae.plt_digit(xhat_vamp[iplt,:])
        if iplt == 0:
            plt.title("ML-VAMP",fontsize=fontsize)
                

    # Create figure
    #plt.show()
    plt.savefig("mnist_inpaint.png")
    print("Images stored in mnist_inpaint.png")