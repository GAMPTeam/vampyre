"""
inpaint.py:  Methods for MNIST inpainting
"""

from __future__ import division
from __future__ import print_function

import vae
import tensorflow as tf
import numpy as np
import pickle
import scipy.special

import os
import sys
vp_path = os.path.abspath('../../')
if not vp_path in sys.path:
    sys.path.append(vp_path)
import vampyre as vp

class InpaintMeth:
    def __init__(self,xtrue,param_fn='param.p',erase_pix0=280, erase_pix1=560):
        """
        Base class for VAE-based inpainting.
        
        Each specific method derives from this class
        
        :param xtrue:  Images of pixels to reconstruct
        :param param_fn:  File name for the saved VAE parameters
        :param erase_pix0:  First pixel to erase
        :param erase_pix1:  Lasst pixel to erase                
        """        
        
        # Load the data from the parameter file
        self.param_fn = param_fn
        
        # True images
        self.xtrue = xtrue
                
        # Number of images
        self.batch_size=xtrue.shape[0]
        
        # Pixels to erase
        self.erase_pix0 = erase_pix0
        self.erase_pix1 = erase_pix1
        
    def compute_loss(self):
        """
        Compute the loss function of the estimate
        """
        # Compute the loss on the prior
        zhat0 = self.zhat0        
        loss_prior = 0.5*np.mean(np.sum(zhat0**2,axis=1), axis=0)
        
        # Get the indices that are not erases
        npix = self.xtrue.shape[1]
        Ikeep = np.hstack((range(self.erase_pix0), range(self.erase_pix1,npix)))
        
        # Compute the prediction loss on the non-erased pixels
        xhat_logit = self.xhat_logit[:,Ikeep]
        xlabel = self.xtrue[:,Ikeep]        
        loss_pred = np.maximum(xhat_logit,0) - xhat_logit * xlabel +\
            np.log(1 + np.exp(-np.abs(xhat_logit)))
        loss_pred = np.mean(np.sum(loss_pred,axis=1),axis=0)
        
        # Compute the final loss
        loss = loss_pred + loss_prior
        return loss
        
class MapInpaint(InpaintMeth):
    def __init__(self,xtrue,param_fn='param.p', erase_pix0=280,\
        erase_pix1=560, n_steps=1000, recon_mode='mmse',\
        nsteps_init=500, lr_adam=0.01, lr_sgd=0.01,nit_burn_in=2000):
        """
        Inpainting method based on the VAE-based likelihood.  Two methods are 
        supported.
        
        If :code:`recon_mode=='map'`, then the method finds the MAP estimate
        using numerical minimization of the negative log likelihood.
        If :code:`recon_mode=='mmse'`, then the method finds the MMSE estimate
        using Langevin sampling.
        
        :param nit_burn_in:  For MMSE estimation, this is the number of initial 
           optimization steps that are ignored for burn in.
        :param n_steps:  number of optimizatoin iterations
        """        
        InpaintMeth.__init__(self,xtrue,param_fn,erase_pix0,erase_pix1)
        self.n_steps = n_steps
        self.recon_mode = recon_mode
        self.nsteps_init = nsteps_init
        self.lr_adam = lr_adam
        self.lr_sgd = lr_sgd
        self.nit_burn_in = nit_burn_in
    
        
    def reconstruct(self):
        """
        Reconstruct erase digits
        """
        # Build the VAE using the previously trained parameters
        vae_net = vae.VAE(n_steps=self.n_steps, \
            erase_pix0=self.erase_pix0, erase_pix1=self.erase_pix1,\
            mode='recon',param_fn=self.param_fn,\
            recon_mode=self.recon_mode,nsteps_init=self.nsteps_init,\
            lr_adam=self.lr_adam, lr_sgd=self.lr_sgd)
        vae_net.build_graph()
        
        # Save the network
        self.vae_net = vae_net
        
        # Run the reconstruction optimization
        vae_net.reconstruct(self.xtrue,restore=False)
        
        if self.recon_mode == 'mmse':
            """
            For MMSE reconstruction, we compute the values from the averages
            of the samples 
            """
            zsamp_hist = np.array(vae_net.hist_dict['zsamp'])
            xhat_hist = np.array(vae_net.hist_dict['xhat'])
            nit = zsamp_hist.shape[0]

            print("Compute means and variances")            
            I = range(self.nit_burn_in,nit)            
            self.zhat0 = np.mean(zsamp_hist[I,:,:],axis=0)
            self.xhat = np.mean(xhat_hist[I,:,:],axis=0)
            self.zhat0_var = np.std(zsamp_hist[I,:,:],axis=0)**2
            self.xhat_var = np.std(xhat_hist[I,:,:],axis=0)**2
            
        else: 
            """
            For MAP reconstruction, we get the values in the Tensorflow
            graph
            """            
            # Get the results
            with tf.Session() as sess:
                vae_net.restore(sess)
                [self.zhat0, self.xhat_logit, self.xhat] = sess.run(\
                    [vae_net.z_samp, vae_net.xhat_logit, vae_net.xhat])
                [self.loss_vals, self.loss_slice, self.loss_prior, self.pred_err, self.loss] =\
                    sess.run([vae_net.loss_vals, vae_net.loss_slice, vae_net.loss_prior, \
                        vae_net.pred_err, vae_net.loss], feed_dict={vae_net.x: self.xtrue})
            self.xhat_var = []
            self.zhat0_var = []
                    
        
            
class VAMPInpaint(InpaintMeth):
    def __init__(self,xtrue,param_fn='param.p', erase_pix0=280,\
        erase_pix1=560, n_steps=200,map_est=False,admm=False):
        """
        VAMP-based inpainting
        
        :param n_steps:  number of optimizatoin iterations
        :param admm: use ADMM solver instead of regular ML-VAMP solver.
        :param map_est:  Perform MAP estimation (default is MMSE)
        """        
        InpaintMeth.__init__(self,xtrue,param_fn,erase_pix0,erase_pix1)
        self.n_steps = n_steps 
        self.map_est = map_est
        self.admm = admm
        
    def reconstruct(self):
        """
        Reconstruct the images
        """
        self.use_fix_var = False
        
        # Get dimensions
        batch_size = self.xtrue.shape[0]
        npix = self.xtrue.shape[1]
        Ierase = range(self.erase_pix0,self.erase_pix1)
        Ikeep = np.setdiff1d(range(npix), Ierase)
        y = self.xtrue[:,Ikeep].T
                

        # Find equivalent variance so that logistic output is roughly 
        # equivalent to a probit
        wvar = logistic_var()
                
        # Construct the first layer which is a Gaussian prior
        with open(self.param_fn, "rb") as fp:
            [Wdec,bdec,Wenc,benc] = pickle.load(fp)
        
        n0 = Wdec[0].shape[0]
        est0 = vp.estim.GaussEst(0,1,shape=(n0,batch_size))
        est_list = [est0]
        
        # To improve the robustness, we add damping and a small boost in the variance
        if self.map_est:
            damp = 0.25
        else:
            damp = 0.75
        damp_var = 0.5
        alpha_max = 1-1e-3     
        
        # Loop over layers in the decoder model
        nlayers = len(Wdec)
        
        msg_hdl_list = []
        for i in range(nlayers):
            # Get matrices for the layer
            Wi = Wdec[i].T
            bi = bdec[i]
                
            # On the final layer, perform the erasing and add noise
            wvari = 0
            if (i == nlayers-1):
                Wi = Wi[Ikeep,:]
                bi = bi[Ikeep]
                if not self.map_est:
                    wvari = wvar            
            
            n1,n0 = Wi.shape
            zshape0 = (n0,batch_size)
            zshape1 = (n1,batch_size)
                        
            Wiop = vp.trans.MatrixLT(Wi,zshape0)
            esti = vp.estim.LinEstimTwo(Wiop,bi[:,None],wvari)
            est_list.append(esti)
            
            # Add the nonlinear layer
            if (i < nlayers-1):
                # For all but the last layer, this is a ReLU
                esti = vp.estim.ReLUEstim(zshape1,map_est=self.map_est)
            elif self.map_est:
                # Final layer, MAP case
                esti = vp.estim.LogisticEstim(y,rvar_init=100,gtol=1e-6)
            else:                
                # Final layer, MMSE case
                esti = vp.estim.HardThreshEst(y,zshape1)
            est_list.append(esti)
        
            # Add the message handlers 
            rvarmin = 0.01  
            msg_hdl0 = vp.estim.MsgHdlSimp(shape=zshape0,damp=damp,\
                damp_var=damp_var,alpha_max=alpha_max,rep_axes=(0,),\
                rvar_min=rvarmin)
            msg_hdl1 = vp.estim.MsgHdlSimp(shape=zshape1,damp=damp,\
                damp_var=damp_var,alpha_max=alpha_max,rep_axes=(0,),\
                rvar_min=rvarmin)                      
            msg_hdl_list.append(msg_hdl0)            
            msg_hdl_list.append(msg_hdl1)
            

        msg_hdl_list[1].alpha_max = 0.95
                        
        # Create the MLVamp solver
        solver = vp.solver.MLVamp(est_list,msg_hdl_list,comp_cost=True,\
                hist_list=['zhat','zhatvar','rvarfwd','rvarrev','cost'],\
                nit=self.n_steps, prt_period=10)
        self.solver = solver
                
        # Run the solver
        solver.solve()                    

        # Get final estimate                    
        zhat = solver.zhat
        Wi = Wdec[nlayers-1].T
        bi = bdec[nlayers-1]
        zfinal = Wi.dot(zhat[-2]) + bi[:,None]
        
        # Get final variance
        self.zhatvar = np.array(solver.hist_dict['zhatvar'])        
        
        self.zhat0 = zhat[0].T
        self.xhat_logit = zfinal.T
        self.xhat = 1/(1+np.exp(-self.xhat_logit))
        
        
def logistic_var():
    """
    Finds a variance to match probit and logistic regression.
    
    Finds a variance :math:`\\tau_w` such that,
    
        :math:`p=P(W < z) \\approx \\frac{1}{1+e^{-z}},`
        
    where :math:`W \\sim {\\mathcal N}(0,\\tau_w)`.
    """
    z = np.linspace(-5,5,1000)      # z points to test
    p1 = 1/(1+np.exp(-z))           # target probability
    var_test = np.linspace(2,3,1000)
    err = []
    for v in var_test:
        p2 = 0.5*(1+scipy.special.erf(z/np.sqrt(v*2)))
        err.append(np.mean((p1-p2)**2))
    
    i = np.argmin(err)
    wvar = var_test[i]
    return wvar            

