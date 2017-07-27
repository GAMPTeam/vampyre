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

class MLVampAdmm(Solver):
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
        self.zhatfwd = [None]*(nlayers-1)
        self.zhatrev = [None]*(nlayers-1)
        self.zhatvarfwd = [None]*(nlayers-1)
        self.zhatvarrev = [None]*(nlayers-1)
        self.zhat = []
        self.zhatvar = []
        self.rfwd = []
        self.rvarfwd = []
        self.rrev = []
        self.rvarrev = []
        self.sfwd = []
        self.srev = []
        
        # Cost terms
        self.msg_cost = np.zeros((nlayers-1,2))
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
        self.sfwd.append(np.zeros(z0.shape))
        self.srev.append(np.zeros(z0.shape))
        self.it = 0
        
        
        # Loop over the middle nodes
        for i in range(1,nlayers-1):
            # Run the initial estimator for node i
            esti = self.est_list[i]
            zi,zvari = esti.est_init()
            
            # Extract the two components and set them as the 
            # forward and reverse messages
            zi0, zi1 = zi
            zvari0, zvari1 = zvari
            self.sfwd.append(np.zeros(zi1.shape))
            self.srev.append(np.zeros(zi1.shape))
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
                        
        # Set the initial messages.  Right now, we only set variances
        for i in range(nlayers-1):
            r, rvar = self.msg_hdl_list[i].msg_init(0)
            if not rvar is None:                
                self.rvarfwd[i] = rvar
            r, rvar = self.msg_hdl_list[i].msg_init(1)
            if not rvar is None:                
                self.rvarrev[i] = rvar            
                
        # Initialize the history
        self.init_hist()
        
    def min_primal_avg(self):
        """
        Minimizes the primal average variables, zhat
        """
        nlayers = len(self.est_list)
        self.fgrad = np.zeros(nlayers-1)
        
        for i in range(nlayers-1):
            
            msg_hdl = self.msg_hdl_list[i]
            rvarrev = common.repeat_axes(self.rvarrev[i],msg_hdl.shape,\
                msg_hdl.rep_axes,rep=False)
            rvarfwd = common.repeat_axes(self.rvarfwd[i],msg_hdl.shape,\
                msg_hdl.rep_axes,rep=False)
            self.zhat[i] = (rvarrev*(self.zhatrev[i]-self.sfwd[i]) +\
                rvarfwd*(self.zhatfwd[i]-self.srev[i]))/(rvarfwd + rvarrev)
                
            # Compute the gradients
            grad0 = (self.zhatrev[i] - self.rfwd[i])/rvarfwd
            grad1 = (self.zhatfwd[i] - self.rrev[i])/rvarrev
            self.fgrad[i] = np.mean((grad0+grad1)**2)
            #self.fgrad[i,1] = np.mean(grad1**2)
            
                
    def min_primal_fwd_rev(self):
        """
        Minimizes the primal variables, zhatrev and zhatfwd
        """              
        nlayers = len(self.est_list)
        
        for i in range(nlayers):
            
            # Get estimate for layer i
            esti = self.est_list[i]
            
            if i==0:
                # Initial layer takes only the reverse message
                self.rrev[0] = self.srev[0] + self.zhat[0]
                ri = self.rrev[0]
                rvari = self.rvarrev[0]
                if self.comp_cost:
                    zi, zvari, ci = esti.est(ri,rvari,return_cost=True)
                    self.node_cost[i] = ci
                else:
                    zi, zvari = esti.est(ri,rvari,return_cost=False)
                self.zhatfwd[0] = zi
                self.zhatvarfwd[0] = zvari
            elif i==nlayers-1:
                # Final layer only takes the forward message
                self.rfwd[i-1] = self.sfwd[i-1] + self.zhat[i-1]
                ri = self.rfwd[i-1]
                rvari = self.rvarfwd[i-1]
                if self.comp_cost:
                    zi, zvari, ci = esti.est(ri,rvari,return_cost=True)
                    self.node_cost[i] = ci
                else:
                    zi, zvari = esti.est(ri,rvari,return_cost=False)
                self.zhatrev[i-1] = zi
                self.zhatvarrev[i-1] = zvari
            else:
                # Middle layers take the forward and reverse messages
                self.rfwd[i-1] = self.sfwd[i-1] + self.zhat[i-1]
                self.rrev[i] = self.srev[i] + self.zhat[i]
                ri = [self.rfwd[i-1], self.rrev[i]]
                rvari = [self.rvarfwd[i-1], self.rvarrev[i]]
                if self.comp_cost:
                    zi, zvari, ci = esti.est(ri,rvari,return_cost=True)
                    self.node_cost[i] = ci
                else:
                    zi, zvari = esti.est(ri,rvari,return_cost=False)
                                    
                # Unpack the estimates and extract the forward estimate
                zi0, zi1 = zi
                zvari0, zvari1 = zvari
                self.zhatrev[i-1] = zi0
                self.zhatvarrev[i-1] = zvari0
                self.zhatfwd[i] = zi1
                self.zhatvarfwd[i] = zvari1                            
                
    def dual_update(self):
        """
        Udpates the dual variables
        """
        
        nlayers = len(self.est_list)
        
        self.con = np.zeros((nlayers-1,2))
        
        for i in range(nlayers-1):        
            # Get the message handler
            msg_hdl = self.msg_hdl_list[i]
            
            # Update the dual parameters
            damp = msg_hdl.damp
            self.sfwd[i] += damp*(self.zhat[i]-self.zhatrev[i])
            self.srev[i] += damp*(self.zhat[i]-self.zhatfwd[i])
            
            # Compute the message costs
            self.msg_cost[i,0] = msg_hdl.cost(\
                self.zhatrev[i],self.zhatvarrev[i],self.rfwd[i],self.rvarfwd[i])
            self.msg_cost[i,1] = msg_hdl.cost(\
                self.zhatfwd[i],self.zhatvarfwd[i],self.rrev[i],self.rvarrev[i])
            
            # Compute the average constraint values
            self.con[i,0] = np.mean((self.zhat[i]-self.zhatrev[i])**2)
            self.con[i,1] = np.mean((self.zhat[i]-self.zhatfwd[i])**2)

            # Update the variances
            damp_var = msg_hdl.damp_var   
            self.it += 1
            if self.it < 1000:
                gamfwd = (1-damp_var)/self.rvarfwd[i] +\
                    damp_var*(1/self.zhatvarfwd[i] - 1/self.rvarrev[i])
                self.sfwd[i] *= 1/gamfwd/self.rvarfwd[i]
                self.rvarfwd[i] = 1/gamfwd*msg_hdl.var_scale
                gamrev = (1-damp_var)/self.rvarrev[i] +\
                    damp_var*(1/self.zhatvarrev[i] - 1/self.rvarfwd[i])
                self.sfwd[i] *= 1/gamrev/self.rvarrev[i]                    
                self.rvarrev[i] = 1/gamrev*msg_hdl.var_scale
                
                
            if 0:
                for j in [0,1]:
                    if self.con[i,j] > self.fgrad[i,j]:
                        scale = 1/(1+damp_var)
                    else:
                        scale = (1+damp_var)
                    if j==0:
                        self.rvarfwd[i] *= scale
                    else:
                        self.rvarrev[i] = scale*self.rvarrev[i]
                        
        
    def add_cost(self):
        """
        Computes the total cost from the node and message costs
        """        
        if self.comp_cost:            
            self.cost = np.sum(self.node_cost) - np.sum(self.msg_cost)\
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
            self.min_primal_fwd_rev()
            self.min_primal_avg()
            self.dual_update()
            self.add_cost()
            self.save_hist()
            
            if self.prt_period > 0:
                if (it % self.prt_period == 0):
                    print("it={0:4d} cost={1:12.4e} con={2:12.4e} fgrad={3:12.4e}".format(\
                        it, self.cost, np.mean(self.con), np.mean(self.fgrad)))

        """
        for it in range(self.nit):
            if self.prt_period > 0:
                if (it % self.prt_period == 0):
                    print("it={0:d}".format(it))
            self.fwd_msg() 
            self.rev_msg()
            self.add_cost()
        """ 
            

