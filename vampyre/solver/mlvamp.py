"""
mlvamp.py:  Multi-layer VAMP solver and test routines
"""
from __future__ import division

# Import general packages
import numpy as np

# Import other subpackages
import vampyre.common as common
import vampyre.estim as estim
import vampyre.trans as trans

# Import methods and classes from the same vampyre sub-package
from vampyre.solver.base import Solver

class MLVamp(Solver):
    """
    Mulit-layer VAMP solver
    
    The VAMP solver performs estimation on a penalty function of the form:
    
    :math:`f(z) = f_0(z_0) + \\sum_\\ell=1^{L-1} f_\\ell(z_\ell,z_{\\ell+1}) + f_L(z_L)`
    
    where :math:`f_\\ell(\\cdot)` are the estimators for each 'layer'.
    
    :param est_list:  List of estimators for the variables.  There should be
       :math:`L+1` estimators in the list
    :param msg_hdl_list:  List of message handlers where 
        :code:`msg_hdl_list[i]` is used for any message incoming to variable
        :math:`z_i`.  There should be :math:`L` elements in the list.
    :param hist_list:  List of attributes to save the history of during the
       course of the iterations.
    :param nit:  Maximum number of iterations          
    :param comp_cost:  Compute cost
    """
    def __init__(self, est_list, msg_hdl_list=[], hist_list=[], nit=10,\
        comp_cost=False):
        Solver.__init__(self,hist_list)
        self.est_list = est_list
        self.msg_hdl_list = msg_hdl_list
        self.nit = nit
        self.comp_cost = comp_cost
        
        # Check if all estimators can compute the cost
        nlayers = len(self.est_list)
        for i in range(nlayers):
            esti = self.est_list[i]
            if self.comp_cost and not esti.cost_avail:
                errstr = "Requested cost computation, but cost_avail==False"\
                    + " for estimator " + str(i)
                raise common.VpException(errstr)
            self.comp_cost = self.comp_cost and esti.cost_avail
        
    def init_msg(self):
        """
        Compute the initial estimates and message.  If there are nlayers:
        
        zhat[i],zhatvar[i] = mean and variance from node i, i=0,...,nlayers-1
        rfwd[i],rvarfwd[i] = forward message from i to i+1,  i=0,...,nlayers-2
        rrev[i],rvarrev[i] = reverse message from i+1 to i,  i=0,...,nlayers-2        
        """
 
        # Initialize lists 
        nlayers = len(self.est_list)
        self.zhat = []
        self.zhatvar = []
        self.rfwd = []
        self.rvarfwd = []
        self.rrev = []
        self.rvarrev = []
        
        # Cost terms
        self.msg_cost_fwd = np.zeros(nlayers-1)
        self.msg_cost_rev = np.zeros(nlayers-1)
        self.node_cost = np.zeros(nlayers)
        self.Hgauss = np.zeros(nlayers-1)
        self.cost = 0

        # Run the initial estimate at node 0
        est0 = self.est_list[0]
        z0, zvar0 = est0.est_init()
        self.zhat.append(z0)
        self.zhatvar.append(zvar0)
        self.rfwd.append(z0)
        self.rvarfwd.append(zvar0)
        
        
        # Loop over the middle nodes
        for i in range(1,nlayers-1):
            # Run the initial estimator for node i
            esti = self.est_list[i]
            zi,zvari = esti.est_init()
            
            # Extract the two components and set them as the 
            # forward and reverse messages
            zi0, zi1 = zi
            zvari0, zvari1 = zvari
            self.zhat.append(zi1)
            self.zhatvar.append(zvari1)
            self.rfwd.append(zi1)
            self.rvarfwd.append(zvari1)    
            self.rrev.append(zi0)
            self.rvarrev.append(zvari0)
            
        # Run the last node and save the output in the reverse message
        est = self.est_list[nlayers-1]
        ri, rvari = est.est_init()
        self.rrev.append(ri)
        self.rvarrev.append(rvari)    
        
        # Initialize the error vectors
        self.zerr = np.zeros(nlayers-1)
        self.rerr = np.zeros(nlayers-1)
        
        # Initialize the history
        self.init_hist()
        
    def fwd_msg(self):
        """
        Forward message passing sequence.
        
        This processes nodes 0 to nlayers-2.
        """
        nlayers = len(self.est_list)
        return_cost = False
        
        for i in range(0,nlayers-1):
            
            # Get estimate for layer i
            esti = self.est_list[i]
            if i==0:
                # Initial layer takes only the reverse message
                ri = self.rrev[0]
                rvari = self.rvarrev[0]
                if self.comp_cost:
                    zi, zvari, ci = esti.est(ri,rvari,return_cost=True)
                    self.node_cost[i] = ci
                else:
                    zi, zvari = esti.est(ri,rvari,return_cost=False)
                self.zhat[0] = zi
                self.zhatvar[0] = zvari
            else:
                # Middle layers take the forward and reverse messages
                ri = [self.rfwd[i-1], self.rrev[i]]
                rvari = [self.rvarfwd[i-1], self.rvarrev[i]]
                zi,zvari = esti.est(ri,rvari,return_cost)
            
                # Unpack the estimates and extract the forward estimate
                zi0, zi1 = zi
                zvari0, zvari1 = zvari
                self.zhat[i] = zi1
                self.zhatvar[i] = zvari1
            
            
            # Compute forward message
            msg_hdl = self.msg_hdl_list[i]
            self.rfwd[i], self.rvarfwd[i] = msg_hdl.msg_sub(\
                self.zhat[i],self.zhatvar[i],self.rrev[i],self.rvarrev[i],\
                self.rfwd[i], self.rvarfwd[i])
                
            # Compute forward message cost and Gaussian entropy 
            if self.comp_cost:
                self.msg_cost_fwd[i] = msg_hdl.cost(\
                    self.zhat[i],self.zhatvar[i],self.rrev[i],self.rvarrev[i])
                self.Hgauss[i] = msg_hdl.Hgauss(self.zhatvar[i])
                
            
        # Save results. We copy rvarfwd into rvar so that the same variable
        # can be used for rvarfwd and rvarrev
        self.rvar = self.rvarfwd
        self.save_hist()
    
  
    def rev_msg(self):
        """
        Reverse message passing sequence.
        
        This pass processes nlayer-1 to 1.
        """
        nlayers = len(self.est_list)
        return_cost = False
        
        for i in range(nlayers-2,-1,-1):
            
            # Compute the estimate at node i+1
            esti = self.est_list[i+1]
            if i == nlayers-2:
                # Use only the forward message
                ri = self.rfwd[i]
                rvari = self.rvarfwd[i]
                if self.comp_cost:
                    zi,zvari,ci = esti.est(ri,rvari,return_cost=True)
                    self.node_cost[i+1] = ci
                else:
                    zi,zvari = esti.est(ri,rvari,return_cost=False)
                
                self.zhat[i] = zi
                self.zhatvar[i] = zvari
            else:
                # Use forward and reverse messages                
                ri = [self.rfwd[i], self.rrev[i+1]]
                rvari = [self.rvarfwd[i], self.rvarrev[i+1]]
                zi,zvari = esti.est(ri,rvari,return_cost)
            
                # Unpack the estimates and extract the reverse estimate
                zi0, zi1 = zi
                zvari0, zvari1 = zvari
                self.zhat[i] = zi0
                self.zhatvar[i] = zvari0
            
            # Compute reverse message
            msg_hdl = self.msg_hdl_list[i]
            self.rrev[i], self.rvarrev[i] = msg_hdl.msg_sub(\
                self.zhat[i],self.zhatvar[i],self.rfwd[i],self.rvarfwd[i],\
                self.rrev[i], self.rvarrev[i])
                
            # Compute reverse message cost and Gaussian entropy 
            if self.comp_cost:
                self.msg_cost_rev[i] = msg_hdl.cost(\
                    self.zhat[i],self.zhatvar[i],self.rfwd[i],self.rvarfwd[i])
                self.Hgauss[i] = msg_hdl.Hgauss(self.zhatvar[i])
                                                 
                    
        # Save results. We copy rvarrev into rvar so that the same variable
        # can be used for rvarfwd and rvarrev
        self.rvar = self.rvarrev
        self.save_hist()
        
    def add_cost(self):
        """
        Computes the total cost from the node and message costs
        """        
        if self.comp_cost:            
            self.cost = np.sum(self.node_cost) - np.sum(self.msg_cost_fwd)\
               - np.sum(self.msg_cost_rev) + np.sum(self.Hgauss)
               
               
    def solve(self,init=True):
        """
        Main iterative solving routine using the forward-backward algorithm.
        
        :param Boolean init:  Set to initialize.  Otherwise, the solver
           will run where it left off last time.
        """
        if init:
            self.init_msg()

        for it in range(self.nit):
            self.fwd_msg()
            self.rev_msg()
            self.add_cost()
            

