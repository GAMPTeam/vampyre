# -*- coding: utf-8 -*-
"""
randmlp:  Random MLP class definitions

@author: Sundeep
"""
import numpy as np
import pickle

"""
Randomly generated multilayer perceptron 
"""
class RandomMLP:
    
    """
    Constructor
    """
    def __init__(self, nin, nhid, nout, sparse_tgt=[]):
        # Get dimensions
        self.nin  = nin
        self.nhid = nhid
        self.nout = nout
        self.nlayers = len(nhid)
        
        # Sparsity target to adjust bias levels in each layer
        if sparse_tgt is None:
            self.sparse_tgt = 0.4*np.ones(self.nlayers)
        else:
            self.sparse_tgt = sparse_tgt
        
        # Number of samples used in calibrating the parameters
        self.ns = 100
        
        # Pickle file name 
        self.save_file = 'mlp.p'
   
    """
    Saves the weights to a pickle file
    """  
    def save_weigths(self):
        pickle.dump((self.Ws,self.bs), open(self.save_file, "wb") )

    """
    Restore weights
    """  
    def restore_weigths(self):
        self.Ws, self.bs = pickle.load(open(self.save_file, "rb") )
        
    """
    Generate random weights based on sparsity in each layer
    """    
    def gen_weigths(self):

        # Create list to store weights and biases
        self.Ws = []
        self.bs = []
        self.z0s = []
        self.z1s = []
        
        # Generate random input
        x = np.random.randn(self.nin,self.ns)
        z0 = x

        for i in range(self.nlayers+1):
            
            self.z0s.append(z0)
            # Get dimensions for the layer
            n0 = z0.shape[0]    
            if i==self.nlayers:
                n1 = self.nout
                stgt = 1
            else:
                n1 = self.nhid[i]
                stgt = self.sparse_tgt[i]
            
               
            # Generate linear outputs w/o bias
            z0norm = np.mean(np.abs(z0)**2)
            W = np.random.randn(n1,n0)/np.sqrt(n0*z0norm)           
            z1 = W.dot(z0)
            
            # Sort to find the biases that adjust the correct sparsity 
            # level
            if stgt < 1:
                zsort = np.sort(z1,axis=1)
                itgt = int((1-stgt)*self.ns)
                b = -zsort[:,itgt]
            else:
                b = np.random.randn(n1)
            z1 = z1 + b[:,None]
            
            # Apply the ReLU for the next layer
            z0 = np.maximum(0, z1)
                
            # Save the weights and biases
            self.Ws.append(W)
            self.bs.append(b)            
            self.z1s.append(z1)
            
    """
    Generate the outputs given input x
    """
    def run(self, z0=[], ns=10):
        
        # Generate initial random initial states if they are unspecified
        if z0 == []:
            z0 = np.random.randn(self.nin,ns)
            
        # Lists to store intermediate variables
        zs = []        
        
        # Loop over layers
        for i in range(self.nlayers+1):
            
            # Save input 
            zs.append(z0)
            
            # Linear weights
            W = self.Ws[i]
            b = self.bs[i]
            z1 = W.dot(z0)+b[:,None]
            
            # Save ouptut
            zs.append(z1)
            
            # ReLU for the next layer     
            z0 = np.maximum(0, z1)
        return zs
