"""
vae_train.py:  Trains the VAE using the MNIST dataset.
"""
from __future__ import division
from __future__ import print_function

import vae

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import argparse

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Trains a VAE model for the MNIST data')
parser.add_argument('-nsteps',action='store',default=20000,type=int,\
    help='total number of steps')
parser.add_argument('-param_fn',action='store',default='param.p',type=str,\
    help='file name for the parameter file')
parser.add_argument('-restore', dest='restore', action='store_true',\
    help="Continue from previous run")    
parser.set_defaults(restore=False)        

args = parser.parse_args()
nsteps = args.nsteps
restore = args.restore
param_fn = args.param_fn

# Dimensions of the layers        
enc_dim = [784,400,20]
dec_dim = [20,400,784]

# Load MNIST
if not 'mnist' in locals():
    mnist = input_data.read_data_sets('MNIST')
    
# Build the VAE
#vae_net = vae.VAE(enc_dim, dec_dim, n_steps=int(20000))
vae_net = vae.VAE(enc_dim, dec_dim, n_steps=int(nsteps))
vae_net.build_graph()

# Train the model
vae_net.train(mnist,restore=restore)

# Dump the matrices
with tf.Session() as sess:
    vae_net.dump_matrices(sess, 'param.p')
print("Data stored in file "+param_fn)

    
