"""
mlvamp.py:  Multi-layer VAMP solver and test routines
"""
from __future__ import division

# Import general packages
import numpy as np
import time

# Import other subpackages
import vampyre.common as common

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
    :param prt_period:  Print summary every :code:`prt_period` iterations.
       When :code:`prt_period==0`, there is no printing.
    :param rvar_fix:  Fixed variances.  Value of :code:`None` means the 
       variances are updated adaptively.
    """
    def __init__(self, est_list, msg_hdl_list=[], hist_list=[], nit=10,\
        comp_cost=False,prt_period=0):
        Solver.__init__(self,hist_list)
        self.est_list = est_list
        self.msg_hdl_list = msg_hdl_list
        self.nit = nit
        self.comp_cost = comp_cost
        self.prt_period = prt_period
        self.time_iter = 0  # Computation time for last iteration
        
        # Check dimensions
        nlayers = len(self.est_list)
        if self.est_list[0].nvars != 1:
            raise ValueError('First estimator must take 1 variable')
        if self.est_list[-1].nvars != 1:
            raise ValueError('Last estimator must take 1 variable')
        for i in range(0,nlayers-1):
            msg_hdl_shape = self.msg_hdl_list[i].shape
            msg_hdl_var_axes = self.msg_hdl_list[i].var_axes
            if i > 0:
                if (self.est_list[i].nvars != 2):
                    errstr = 'Estimator %s must take 2 variables'\
                        % self.est_list[i].name
                    raise ValueError(errstr)

            if i==0:
                shape0 = self.est_list[i].shape
                var_axes0 = self.est_list[i].var_axes
            else:
                shape0 = self.est_list[i].shape[1]
                var_axes0 = self.est_list[i].var_axes[1]
            if i==nlayers-2:
                shape1 = self.est_list[i+1].shape
                var_axes1 = self.est_list[i+1].var_axes
            else:
                shape1 = self.est_list[i+1].shape[0]
                var_axes1 = self.est_list[i+1].var_axes[0]
            
            if shape0 != shape1:
                errstr = 'Est %s shape %s does not match est %s shape %s'\
                    % (est_list[i].name, str(shape0), est_list[i+1].name, str(shape1))
                raise ValueError(errstr)
            if shape0 != msg_hdl_shape:
                errstr = 'Est %s shape %s does not match msg_hdl shape %s'\
                    % (est_list[i].name, str(shape0), str(msg_hdl_shape))
                raise ValueError(errstr)
                
            if var_axes0 != var_axes1:
                errstr = 'Est %s var_axes %s does not match est %s var_axes %s'\
                    % (est_list[i].name, str(var_axes0), est_list[i+1].name, str(var_axes1))            
            if var_axes0 != msg_hdl_var_axes:
                errstr = 'Est %s var_axes %s does not match msg_hdl var_axes %s'\
                    % (est_list[i].name, str(var_axes0), str(msg_hdl_var_axes))
                raise ValueError(errstr)
            
        
        
        # Check if all estimators can compute the cost
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
        self.var_cost = np.zeros((nlayers-1,2))
        self.node_cost = np.zeros(nlayers)
        self.grad = np.zeros(nlayers-1)
        self.con = np.zeros(nlayers-1)
        self.Hgauss = np.zeros(nlayers-1)
        self.cost = 0

        # Run the initial estimate at node 0
        est0 = self.est_list[0]
        z0, zvar0 = est0.est_init()
        r0, rvar0 = self.msg_hdl_list[0].init_msg(z0,zvar0,idir=1)
        self.zhat.append(z0)
        self.zhatvar.append(zvar0)
        self.rfwd.append(r0)
        self.rvarfwd.append(rvar0)
                
        # Loop over the middle nodes
        for i in range(1,nlayers-1):
            # Run the initial estimator for node i
            esti = self.est_list[i]
            zi,zvari = esti.est_init()
            
            # Extract the two components and set them as the 
            # forward and reverse messages
            zi0, zi1 = zi
            zvari0, zvari1 = zvari
            ri0, rvari0 = self.msg_hdl_list[i-1].init_msg(zi0,zvari0,idir=0)
            ri1, rvari1 = self.msg_hdl_list[i].init_msg(zi1,zvari1,idir=1)            
            self.zhat.append(zi1)
            self.zhatvar.append(zvari1)
            self.rfwd.append(ri1)
            self.rvarfwd.append(rvari1)    
            self.rrev.append(ri0)
            self.rvarrev.append(rvari0)
            
        # Run the last node and save the output in the reverse message
        est = self.est_list[nlayers-1]
        zi0, zvari0 = est.est_init()
        ri0, rvari0 = self.msg_hdl_list[nlayers-2].init_msg(zi0,zvari0,idir=0)
        self.rrev.append(ri0)
        self.rvarrev.append(rvari0)    
        
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
                if self.comp_cost:
                    zi, zvari, ci = esti.est(ri,rvari,return_cost=True,ind_out=[1])
                    self.node_cost[i] = ci
                else:
                    zi, zvari = esti.est(ri,rvari,return_cost=False,ind_out=[1])
                            
                # Unpack the estimates and extract the forward estimate
                zi1 = zi[0]
                zvari1 = zvari[0]
                self.zhat[i] = zi1
                self.zhatvar[i] = zvari1
            
            # Compute forward message
            msg_hdl = self.msg_hdl_list[i]
            self.rfwd[i], self.rvarfwd[i] = msg_hdl.msg_sub(\
                self.zhat[i],self.zhatvar[i],idir=0)            
                
            # Compute forward message cost and Gaussian entropy 
            if self.comp_cost:
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
                zi,zvari = esti.est(ri,rvari,return_cost,ind_out=[0])
            
                # Unpack the estimates and extract the reverse estimate
                zi0 = zi[0]
                zvari0 = zvari[0]
                self.zhat[i] = zi0
                self.zhatvar[i] = zvari0
                
            # Compute reverse message
            msg_hdl = self.msg_hdl_list[i]
            self.rrev[i], self.rvarrev[i] = msg_hdl.msg_sub(\
                self.zhat[i],self.zhatvar[i],idir=1)
                
            # Compute reverse message cost and Gaussian entropy 
            if self.comp_cost:
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
            nvars = len(self.msg_hdl_list)
            for i in range(nvars):
                var_costi, gradi, coni = self.msg_hdl_list[i].get_cost_terms()
                self.var_cost[i,:] = [var_costi[0], var_costi[1]]
                self.grad[i] = gradi
                self.con[i] = coni
            self.cost = np.sum(self.node_cost) - np.sum(self.var_cost)\
               + np.sum(self.Hgauss)
               
               
    def solve(self,init=True):
        """
        Main iterative solving routine using the forward-backward algorithm.
        
        :param Boolean init:  Set to initialize.  Otherwise, the solver
           will run where it left off last time.
        """
        if init:
            self.init_msg()

        for it in range(self.nit):
            if self.prt_period > 0:
                if (it % self.prt_period == 0) or (it == self.nit-1):                    
                    print("it={0:d} cost={1:12.4e} con={2:12.4e} grad={3:12.4e}".format(\
                        it, self.cost, np.mean(self.con), np.mean(self.grad)))
            t0 = time.time()
            self.fwd_msg() 
            self.rev_msg()
            self.add_cost()
            t1 = time.time()
            self.time_iter = t1-t0
            
    def summary(self):
        """
        Prints a summary of the model
        """        
        print('Layer | %-30s | %-10s | %-8s' % ('Estimator', 'shape', 'var_axes'))
        nlayers = len(self.est_list)        
        for i, est in enumerate(self.est_list):
            layer_str = '%s (%s)' % (est.name, est.type_name)
            if i==0:
                print('%3d   | %-30s | %-10s | %-8s' %\
                    (i, layer_str, str(est.shape), str(est.var_axes)))
            elif (i < nlayers-1):
                print('%3d   | %-30s | %-10s | %-8s' %\
                    (i, layer_str, str(est.shape[1]), str(est.var_axes[1])))
            else:
                print('%3d   | %-30s | %-10s | %-8s'  % (i, layer_str, ' ', ' '))
        
            
            

