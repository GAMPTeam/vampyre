"""
vae.py:  Variational autoencoder to develop a multi-layer perceptron

The variational auto-encoder code is based on the implementation by
    https://github.com/y0ast/VAE-TensorFlow
    
For compatibility with VAMP-methods, this uses ReLU activation functions.

This file requires tensorflow installed.
"""
from __future__ import division
from __future__ import print_function
import os.path

# Load packages.
import os
from datetime import datetime
import pickle
import re
import tensorflow as tf
import numpy as np


def plt_digit(x):
    """
    Plots a digit in the MNIST dataset.
    
    :param:`x` is the digit to plot represented as 784 dim vector
    """
    import matplotlib.pyplot as plt
    
    nrow = 28
    ncol = 28
    xsq = x.reshape((nrow,ncol))
    plt.imshow(np.maximum(0,xsq), cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])


class VAE(object):
    def __init__(self, enc_dim=None, dec_dim=None, n_steps=int(1e4), batch_size=100,\
        nstep_prt=50, nstep_save=5000, save_dir='save', loss_type='logistic',\
        mnist_image_summ = True, mode='train', param_fn = 'param.p',\
        recon_save_dir='recon_save', erase_pix0=28*10,\
        erase_pix1=28*20, recon_mode='mmse', nsteps_init=400, lr_adam=0.01,lr_sgd=0.01):
        """
        Variational autoencoder for the MNIST data.
        The class can perform one of two tasks:
        -- When :code:`mode='train'`, the class learns a VAE for the MNIST data.
           The parameters are saved in a parameter file, :code:`param_fn`.
        -- When :code:`mode='recon'`, the class uses the VAE for inpainting 
           reconstruction using the parameters are saved in the parameter file.
           
        The inpainting reconstruction is performed in one of two ways:
        -- MAP reconstruction, which finds the MAP estimate via gradient descent
           on the posterior density.
        -- MMSE reconstruction, which uses Langevin dynamics to sample the 
           posterior density.
        In either case, the samples are stored in the variable :code:`zhat_hist`.
                
        :param enc_dim:  List of dimensions per layer for the encoder
           where :code:`enc_dim[0]` is the input dimension, 
           :code:`enc_dim[i]` is the dimension of hidden layer :code:`i-1` 
           and :code:`enc_dim[-1]` is the latent dimension
        :param enc_dim:  List of dimensions per layer for the decoder, 
            where :code:`dec_dim[0]` is the latent variable dimension 
            and :code:`dec_dim[-1]` is the output dimension
        :param n_steps:  number of steps in training
        :param batch_size:  mini-batch size         
        :param nstep_prt:  number of steps between a print of the training
            progress
        :param nstep_save:  number of steps between a saving
        :param loss_type:  loss type, either 'logistic' or 'relu'
        :param save_dir:  directory to save model        
        :param mnist_image_summ:  Add an image summary for observing the
            MNIST reconstruction.  
        :param mode:  Either :code:`'train'` or :code:`'recon' as described 
            above
        :param recon_mode:  :code:`'map'` for MAP reconstruction, :code:`mmse`
            for generating samples from the posterior density.
        :param param_fn:  filename for pickle parameter file
        :param recon_save_dir:  directory for saving the reconstruction 
            optimization
        :param load_param:  load parameters from the file
        :param erase_pix0:  first index of the erased pixels
        :param erase_pix1:  last index of the erased pixels
        :param lr_adam:  ADAM optimizer learning rate 
        :param lr_sgd:  SGD optimizer learning rate 
        :param nsteps_init:  number of initial steps used before
           starting langevin dynamics
        """
        
        # Training parameters
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.nstep_prt = nstep_prt
        self.nstep_save = nstep_save 
        
        self.loss_type = loss_type
        self.save_dir = save_dir
        self.mnist_image_summ = mnist_image_summ
        self.mode = mode
        self.param_fn = param_fn
        self.recon_save_dir = recon_save_dir
        self.recon_mode = recon_mode
        self.lr_adam = lr_adam
        self.lr_sgd = lr_sgd
        self.nsteps_init = nsteps_init
            
        # Check the mode
        if (mode != 'train') and (mode != 'recon'):
            raise Exception("Parameter \'mode\' must be \'train\' or \'recon\' ")
        if (recon_mode != 'map') and (recon_mode != 'mmse'):
            raise Exception("Parameter \'recon_mode\' must be" +\
                            "\'map\' or \'mmse\' ")

        # Load the parameters for the reconstruction case.  
        # When loading the parameters, the dimensions will be set.   
        if (mode == 'recon'):
            self.load_param()
        else:
            self.enc_dim = enc_dim
            self.dec_dim = dec_dim

        # Set the erase pixels for the reconstruction case
        self.erase_pix0 = erase_pix0
        self.erase_pix1 = erase_pix1
                        
        # Check matching dimensions        
        if self.enc_dim[-1] != self.dec_dim[0]:
            raise Exception("Encoder output dimension does not match "+\
                            "decoder input dimension")
        if self.dec_dim[-1] != self.enc_dim[0]:
            raise Exception("Encoder input dimension does not match "+\
                            "decoder output dimension")     
                            
    def load_param(self):
        """
        Loads previously trained weigths and bias parameters
        """
        with open(self.param_fn, "rb") as fp:
            [Wdec,bdec,Wenc,benc] = pickle.load(fp)

        # Save matrices
        self.Wdec = Wdec
        self.bdec = bdec
        self.Wenc = Wenc
        self.benc = benc        
        
        # Get dimensions        
        nenc = len(Wenc)
        self.enc_dim = np.zeros(nenc+1,dtype=int)
        for i in range(nenc):
            self.enc_dim[i] = Wenc[i].shape[0]
        self.enc_dim[nenc] = Wenc[nenc-1].shape[1]
        
        ndec = len(Wdec)
        self.dec_dim = np.zeros(ndec+1,dtype=int)
        for i in range(ndec):
            self.dec_dim[i] = Wdec[i].shape[0]
        self.dec_dim[nenc] = Wdec[ndec-1].shape[1]
    
        
    def weight_bias_var(self,n0,n1):
        """
        Initial bias and variance
        """
        Winit = tf.truncated_normal([n0,n1], stddev=0.01)
        binit = tf.truncated_normal([n1], stddev=0.01) #tf.constant(0.0,shape=[n1])
        W_var = tf.Variable(Winit,dtype=tf.float32, name='weight')
        b_var = tf.Variable(binit,dtype=tf.float32, name='bias')
        return W_var, b_var
    
    def build_enc(self):    
        """
        Builds the encoder graph.  This is only used in the training case.
        """
                
        # Loop over encoder layers
        nlayers_enc = len(self.enc_dim)-1
        self.Wenc = []
        self.benc = []
        self.zenc = [self.x]
        
        for i in range(nlayers_enc):
            
            layer_name = 'enc{0:d}'.format(i)
            with tf.variable_scope(layer_name):
                n0 = self.enc_dim[i]
                n1 = self.enc_dim[i+1]
                Wi, bi = self.weight_bias_var(n0,n1)
                
                self.Wenc.append(Wi)
                self.benc.append(bi)
                z0 = self.zenc[2*i]
                
                z1 = tf.add(tf.matmul(z0, Wi),bi,name='lin_out')
                self.zenc.append(z1)
                if (i < nlayers_enc-1):
                    z2 = tf.nn.relu(z1,name='relu_out')
                    self.zenc.append(z2)
                    
        # Final encoder output = latent variable
        i = nlayers_enc-1
        self.z_mu = self.zenc[2*i+1]
                    
        # Logvar encoder
        i = nlayers_enc-1
        n0 = self.enc_dim[i]
        n1 = self.enc_dim[i+1]
        z0 = self.zenc[2*i]
        with tf.variable_scope("enc_logvar"):
            Wi, bi = self.weight_bias_var(n0,n1)
            self.Wlogvar = Wi
            self.blogvar = bi
            self.z_logvar = tf.add(tf.matmul(z0, Wi),bi,name='z_logvar')
            
    def build_dec(self):
        """
        Builds the decoder graph.
        
        In the case of reconstruction, the weigths and biases are loaded from 
        the parameter file
        """                     
              
        # Sample latent variable
        with tf.variable_scope("sample"):
            if self.mode == 'recon':
                # For reconstruction, the samples either represent the MAP 
                # estimate or the samples from the posterior
                zinit = np.zeros((self.batch_size,self.dec_dim[0]))
                self.z_samp = tf.Variable(zinit, dtype=tf.float32, name='z_samp')                                    

            else:
                # Random perturbation to latent variable
                # Note that the shape is dynamically set from the shape of z_logvar
                self.epsilon = tf.random_normal(tf.shape(self.z_logvar), name='epsilon')
                
                # Gaussian sampling of the latent variable
                self.z_std = tf.exp(0.5 * self.z_logvar, name="z_std")
                self.z_samp = tf.add(self.z_mu, tf.multiply(self.z_std, self.epsilon),\
                                name="z_samp")

                # Initialize the decoder matrices                                
                self.Wdec = []
                self.bdec = []                         
                            
        # Loop over decoder layer
        nlayers_dec = len(self.dec_dim)-1
        self.zdec = [self.z_samp]
        self.l2_reg = tf.constant(0.0)
        
        for i in range(nlayers_dec):
            
            layer_name = 'dec{0:d}'.format(i)
            with tf.variable_scope(layer_name):
                n0 = self.dec_dim[i]
                n1 = self.dec_dim[i+1]
                if self.mode == 'recon':
                    Wi = tf.constant(self.Wdec[i], name="weight")
                    bi = tf.constant(self.bdec[i], name="bias")
                else:
                    Wi, bi = self.weight_bias_var(n0,n1)
                    self.Wdec.append(Wi)
                    self.bdec.append(bi)

                # Linear step                                
                z0 = self.zdec[2*i]                
                z1 = tf.add(tf.matmul(z0, Wi),bi,name='lin_out')
                self.zdec.append(z1)
                
                # ReLU                
                z2 = tf.nn.relu(z1,name='relu_out')
                self.zdec.append(z2)
                                
        # Get the final output
        self.xhat_logit = self.zdec[-2]
        if self.loss_type == 'logistic':
            self.xhat = tf.nn.sigmoid(self.xhat_logit,name='xhat')
        elif self.loss_type == 'relu':
            self.xhat = tf.nn.sigmoid(2*self.xhat_logit,name='xhat')
        else:
            raise Exception("Unknown loss type "+self.loss_type)
                
    def build_loss_fn(self):
        """
        Adds the ops for the loss functions
        
        The prediction error is computed as follows:
        Let z=xhat_logit
        If loss_type == 'logistic', then 
            -log p(x|z) = -x*log(p) - (1-x)*log(1-p), p=1/(1+e^{-z})
                        = z*(1-x) - log(1+exp(-z))
        If loss_type == 'relu', we take an approximation:
            -log p(x|z) = x*relu(1-z) + (1-x)*relu(1+z)
        """
        with tf.variable_scope("Loss"):
            if self.loss_type == 'logistic':
                self.loss_vals = tf.nn.sigmoid_cross_entropy_with_logits(\
                    logits=self.xhat_logit, labels=self.x)
            elif self.loss_type == 'relu':
                self.loss_vals = \
                    tf.multiply(1-self.x,tf.nn.relu(1+self.xhat_logit)) + \
                    tf.multiply(self.x,  tf.nn.relu(1-self.xhat_logit))
            else:
                raise Exception("Unknown loss_type")
                                
            if self.mode == 'recon':
                # Extract the losses from the visible pixels
                loss_slice0 = tf.slice(self.loss_vals,[0,0],[-1,self.erase_pix0-1])
                loss_slice1 = tf.slice(self.loss_vals,[0,self.erase_pix1],[-1,-1])
                self.loss_slice = tf.concat([loss_slice0, loss_slice1], axis=1)
                
                # Compute the prediction error on the visible pixels
                self.pred_err = tf.reduce_mean(tf.reduce_sum(self.loss_slice,\
                    reduction_indices=1), name="pred_err")
                    
                # Compute the prior loss and total loss
                self.loss_prior = tf.reduce_mean(0.5*tf.reduce_sum(tf.square(self.z_samp), \
                    reduction_indices=1), name="loss_prior")
                self.loss = tf.add(self.pred_err, self.loss_prior, name="loss")                
                
            else:                  
                # In training, we use the VAE loss, which is the prediction error 
                # and the KLD.  See Kigma and Welling.
                self.pred_err = tf.reduce_sum(self.loss_vals,\
                    reduction_indices=1, name="pred_err")
                self.KLD = tf.reduce_sum( -0.5*(1 + self.z_logvar -\
                    tf.square(self.z_mu) - tf.exp(self.z_logvar)),\
                    reduction_indices=1, name="KLD")
                loss0 = tf.reduce_mean(self.KLD+self.pred_err)
                self.loss = tf.add(loss0, self.l2_reg, name="loss")                     
                
            # Loss summary op for tensorboard                
            self.loss_summ = tf.summary.scalar("Loss", self.loss)
                        
            # Create an image summary for the reconstruction.
            # Use this only if the VAE is being used for the MNIST data set
            if self.mnist_image_summ:
                nrow = 28
                ncol = 28
                x_image = tf.reshape(tf.slice(self.x,[0,0],[1,nrow*ncol]), [1,nrow,ncol,1])
                self.x_summ = tf.summary.image("original", x_image)
                xhat_image = tf.reshape(tf.slice(self.xhat,[0,0],[1,nrow*ncol]), [1,nrow,ncol,1])
                self.xhat_summ = tf.summary.image("reconstructed", xhat_image)
                    
        # Add the training ops
        if (self.mode == 'recon') and (self.recon_mode == 'mmse'):
            # Gradient descent optimizer.  
            wvar = 2*self.lr_sgd/self.batch_size
            self.grad_des_step = \
                tf.train.GradientDescentOptimizer(self.lr_sgd).minimize(self.loss) 
            
            # Add dithering for the Langevin dynamics
            self.zsamp_noise = tf.random_normal(self.z_samp.shape, mean=0.0,\
                stddev=np.sqrt(wvar),dtype=tf.float32, name="zsamp_noise")            
            self.zsamp_dither = tf.assign(self.z_samp,\
                tf.add(self.z_samp, self.zsamp_noise), name="zsamp_dither") 

        # Adam optimizer, used for training and MAP estimation
        self.adam_step = tf.train.AdamOptimizer(self.lr_adam).minimize(self.loss)
                                    
       
    def build_graph(self):
        """
        Builds graph
        """
        
        # Clear the grapht
        tf.reset_default_graph()
        
        # Create the placeholder for the input
        nx = self.enc_dim[0]
        self.x = tf.placeholder("float", shape=[None, nx], name='x')


        # Builds the various components        
        if self.mode == 'train':
            self.build_enc()
        self.build_dec()
        self.build_loss_fn()
                    
        # Add the summary op
        self.summary_op = tf.summary.merge_all()
        
        # Create a saver
        self.saver = tf.train.Saver()

        
    def restore(self, sess):
        """
        Restores model from the latest checkpoint
        
        :param sess:  Current tensforflow session 
        """
        if self.mode == 'recon':
            self.save_path = tf.train.latest_checkpoint(self.recon_save_dir)
        else:
            self.save_path = tf.train.latest_checkpoint(self.save_dir)            
        if  self.save_path is None:
            raise Exception("Cannot find a checkpoint file to restore")
        print("Restoring from "+ self.save_path)
        self.saver.restore(sess,  self.save_path)
        
    def create_save_dir(self):
        """
        Creates the save directory
        """
        if self.mode == 'recon':
            if not os.path.exists(self.recon_save_dir):
                os.mkdir(self.recon_save_dir)
        else:            
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            
    def reconstruct(self, xtrue, restore=False):
        """
        Optimization for the reconstruction.  
        
        :param xtrue: MNIST true image
        """                        
        self.create_save_dir()

        # Counter for averaging        
        self.step_avg = 0
                            
        with tf.Session() as sess:
            # Create a log directory name based on the current time.
            # This separates the different runs in tensorboard
            now = datetime.now()
            logdir = "logs" + os.path.sep + now.strftime("%Y%m%d-%H%M%S") 

            # Open the summary writer in the session
            summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)       
            
            if restore:
                # Restore the trained model
                self.restore(sess)
                
                # Extract the last step from the path name
                step_init = int(re.findall(r'\d+',self.save_path)[0])
            else:
                # Initializing
                print("Initializing")
                sess.run(tf.global_variables_initializer())
                step_init = 0

            # Loop over training steps
            for step in range(step_init, step_init+self.n_steps):                
                feed_dict = {self.x: xtrue}
                if (step < self.nsteps_init) or (self.recon_mode == 'map'):
                    _, cur_loss, summary_str = sess.run(\
                        [self.adam_step, self.loss, self.summary_op], \
                        feed_dict=feed_dict)
                else:
                    _, cur_loss, summary_str = sess.run(\
                        [self.grad_des_step, self.loss, self.summary_op], \
                        feed_dict=feed_dict)
                    sess.run([self.zsamp_dither])                    
                            
                if step % self.nstep_prt == 0:      
                    print("Step {0} | Loss: {1}".format(step, cur_loss))
                    summary_writer.add_summary(summary_str, step)

                # Update the running averages of the first and second moments
                xhati, xhat_logiti, zsampi = sess.run(\
                    [self.xhat, self.xhat_logit,self.z_samp])

                if (self.step_avg == 0):
                    self.xhat_mean = xhati
                    self.zhat0_mean = zsampi
                    self.zhat0_sq_mean = zsampi**2
                    self.xhat_sq_mean = xhati**2
                else:
                    t = 1/(self.step_avg+1)
                    self.xhat_mean += t*(xhati-self.xhat_mean)
                    self.xhat_sq_mean = t*((xhati**2)-self.xhat_sq_mean)
                    self.zhat0_mean += t*(zsampi-self.zhat0_mean)
                    self.zhat0_sq_mean += t*((zsampi**2)-self.zhat0_sq_mean)  
                self.step_avg += 1
                                                                        
                # Save the checkpoint files
                if ((step % self.nstep_save == 0) and (step > 0)) or (step == self.n_steps-1):
                    save_path = self.recon_save_dir+os.path.sep+"model.ckpt"
                    self.save_path = self.saver.save(sess, save_path,\
                       global_step=step)
                       
                    # Save the current average values
                    xhat_var = self.xhat_sq_mean - self.xhat_mean**2
                    zhat0_var = self.zhat0_sq_mean - self.zhat0_mean**2
                    self.last_avg = [self.xhat_mean,self.zhat0_mean,\
                            xhat_var, zhat0_var]
                            
                    # Reset the counter to reset the averaging
                    self.step_avg = 0
                       
                    # Save the average values in a file
                    fn = "xhat_{0:d}.p".format(step)
                    xhat_save_path = self.recon_save_dir+os.path.sep+fn
                    with open(xhat_save_path, "wb") as fp:
                        pickle.dump(self.last_avg, fp)
                    
                      
            
    def train(self, mnist, restore=False):
        """
        Train using an MNIST dataset
        
        :param mnist: MNIST data structure from tensorflow
        :param restore:  Flag indicating to use the restore from the last 
            execution
        """                        
        self.create_save_dir()
        
        with tf.Session() as sess:
            # Create a log directory name based on the current time.
            # This separates the different runs in tensorboard
            now = datetime.now()
            logdir = "logs" + os.path.sep + now.strftime("%Y%m%d-%H%M%S") 

            # Open the summary writer in the session
            summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)       
            
            if restore:
                # Restore the trained model
                self.restore(sess)
                
                # Extract the last step from the path name
                step_init = int(re.findall(r'\d+',self.save_path)[0])
            else:
                # Initializing
                print("Initializing")
                sess.run(tf.global_variables_initializer())
                step_init = 0

            # Loop over training steps
            for step in range(step_init, step_init+self.n_steps):
                batch = mnist.train.next_batch(self.batch_size)
                feed_dict = {self.x: batch[0]}
                _, cur_loss, summary_str = sess.run(\
                    [self.train_step, self.loss, self.summary_op], \
                    feed_dict=feed_dict)
                            
                if step % self.nstep_prt == 0:      
                    print("Step {0} | Loss: {1}".format(step, cur_loss))
                    summary_writer.add_summary(summary_str, step)
                                    
#               
                # Save the data
                if (step % self.nstep_save == 0) or (step == self.n_steps-1):
                    save_path = self.save_dir+os.path.sep+"model.ckpt"
                    self.save_path = self.saver.save(sess, save_path,\
                       global_step=step)
                       
    def dump_matrices(self, sess, fn):
        """
        Performs a pickle dump of the encoder and decoder matrices
        
        :param sess:  Current tensorflow session
        :param fn:  Filename of the destination pickle file
        """
        # Extract the decoder matrices
        nlayers_dec = len(self.dec_dim)-1
        Wdec = []
        bdec = []
        for i in range(nlayers_dec):
            Wdeci_op = self.Wdec[i]
            bdeci_op = self.bdec[i]
            Wdeci, bdeci = sess.run([Wdeci_op, bdeci_op])
            Wdec.append(Wdeci)
            bdec.append(bdeci)
            
        # Extract the decoder matrices
        nlayers_enc = len(self.enc_dim)-1
        Wenc = []
        benc = []
        for i in range(nlayers_enc):
            Wenci_op = self.Wenc[i]
            benci_op = self.benc[i]
            Wenci, benci = sess.run([Wenci_op, benci_op])
            Wenc.append(Wenci)
            benc.append(benci)
            
        # Dump the parameters to the pickle file
        pickle.dump([Wdec,bdec,Wenc,benc], open(fn, "wb"))
            


    
    
