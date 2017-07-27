"""
vae_train.py:  Trains the VAE using the MNIST dataset.
"""
from __future__ import division
from __future__ import print_function

import vae

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# Dimensions of the layers        
enc_dim = [784,400,20]
dec_dim = [20,400,784]

# Load MNIST
if not 'mnist' in locals():
    mnist = input_data.read_data_sets('MNIST')
    
# Build the VAE
#vae_net = vae.VAE(enc_dim, dec_dim, n_steps=int(20000))
vae_net = vae.VAE(enc_dim, dec_dim, n_steps=int(100))
vae_net.build_graph()

if 1:        
    vae_net.train(mnist,restore=False)

    
if 0:
    
    with tf.Session() as sess:
        vae_net.restore(sess)
        vae_net.dump_matrices(sess, "param.p")
        
        z_samp = np.random.randn(10,20)
        xhat_logit = sess.run(vae_net.xhat_logit, feed_dict={vae_net.z_samp: z_samp})
        xhat = 1/(1+np.exp(-2*xhat_logit))
        vae.plt_digit(xhat[0,:])
                    
                    
        
        
            
    
    





    
    
