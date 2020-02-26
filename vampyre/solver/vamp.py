"""
vamp.py:  Main VAMP solver
"""
from __future__ import division

# Import general packages
import numpy as np
import time

# Import other subpackages
import vampyre.estim as estim
#import vampyre.trans as trans
import vampyre.common as common

# Import methods and classes from the same vampyre sub-package
from vampyre.solver.base import Solver


class Vamp(Solver):
    """
    Vamp solver
    
    The VAMP solver performs estimation on a penalty function of the form:
    
    :math:`f(z) = f_1(z) + f_2(z)`
    
    where :math:`f_1(z)` and :math:`f_2(z)` are two constintuent estimators.
    
    :param est1:  Estimator corresonding to :math:`f_1(z)`
    :param est2:  Estimator corresonding to :math:`f_2(z)`
    :param msg_hdl:  Message handler.  
    :param hist_list:  List of attributes to save the history of during the
       course of the iterations.
    :param nit:  Maximum number of iterations          
    :param comp_cost:  Compute cost
    :param prt_period:  Period for printing progress (value of 0 indicates
        no printing)
    """
    def __init__(self, est1, est2, msg_hdl, hist_list=[], nit=10,\
        comp_cost=False,prt_period=0, is_complex=False, map_est=False):
        Solver.__init__(self,hist_list)
        self.est1 = est1
        self.est2 = est2
        self.msg_hdl = msg_hdl        
        self.nit = nit
        self.comp_cost = comp_cost
        self.prt_period = prt_period
        
        # Check if dimensions match
        if self.est1.shape != self.est2.shape:
            err_str = '%s shape %s does not match %s shape %s' %\
                (self.est1.name, str(self.est1.shape),\
                 self.est2.name, str(self.est2.shape))                
            raise common.VpException(err_str)
        if self.est1.shape != self.msg_hdl.shape:
            err_str = '%s shape %s does not match msg_hdl shape %s' %\
                (self.est1.name, str(self.est1.shape),\
                 str(self.msg_hdl.shape))                
            raise common.VpException(err_str)
            
        if self.est1.var_axes != self.est2.var_axes:
            err_str = '%s var_axes %s does not match %s var_axes %s' %\
                (self.est1.name, str(self.est1.var_axes),\
                 self.est2.name, str(self.est2.var_axes))
            raise common.VpException(err_str)
        if self.est1.var_axes != self.msg_hdl.var_axes:
            err_str = '%s var_axes %s does not match msg_hdl var_axes %s' %\
                (self.est1.name, str(self.est1.var_axes),\
                 str(self.msg_hdl.var_axes))                
            
        # Set default message handler
        if msg_hdl is None:
            self.msg_hdl = estim.MsgHdlSimp()
        
    def summary(self):
        """
        Prints a summary of the model
        """
        print('Variable:  shape: %s, var_axes: %s'\
            % (str(self.msg_hdl.shape),str(self.msg_hdl.var_axes)))
        print('est0: %s (%s)' % (str(self.est1.name),str(self.est1.type_name)))
        print('est1: %s (%s)' % (str(self.est2.name),str(self.est2.type_name)))
        
                        
    def solve(self):
        """
        Runs the main VAMP algorithm
        
        The final estimates are saved in :code:`r1` and :code:`r2`
        along with variances :code:`rvar1` and :code:`rvar2`
        """
                    
        # Check if cost is available for both estimators
        if not self.est1.cost_avail or not self.est2.cost_avail or self.msg_hdl == []:
            self.comp_cost = False
            
        # Initial estimate from the first factor node
        if self.comp_cost:
            z1, zvar1, cost1 = self.est1.est_init(return_cost=True)
        else:
            z1, zvar1 = self.est1.est_init(return_cost=False) 
            cost1 = 0
        
        # Compute message to the second node
        r2, rvar2 = self.msg_hdl.msg_sub(z1,zvar1,idir=0)
        
        # Overwrite message with pre-set values
        r2, rvar2 = self.msg_hdl.init_msg(r2,rvar2,idir=1)
        
        # Initialize other variables
        self.cost1 = cost1
        self.r2 = r2
        self.rvar2 = rvar2
        self.r1 = []
        self.rvar1 = None
        self.var_cost1 = 0
        self.var_cost2 = 0
        self.cost = 0
                
        for it in range(self.nit):
            
            # Estimator 2            
            t0 = time.time()
            if self.comp_cost:  
                z2, zvar2, cost2 = self.est2.est(r2, rvar2, return_cost=True)               
            else:
                z2, zvar2 = self.est2.est(r2, rvar2, return_cost=False)
                cost2 = 0
            self.z2 = z2
            self.zvar2 = zvar2
            self.cost2 = cost2
                        
            # Msg passing to est 1 (from est2 => idir=1)
            r1, rvar1 = self.msg_hdl.msg_sub(z2,zvar2,idir=1)
            self.r1 = r1
            self.rvar1 = rvar1
            t1 = time.time()
            self.time_est2 = t1-t0
            
            # Estimator 1
            if self.comp_cost:
                z1, zvar1, cost1 = self.est1.est(r1, rvar1, return_cost=True)
            else:
                z1, zvar1 = self.est1.est(r1, rvar1, return_cost=False)
                cost1 = 0
            self.z1 = z1
            self.zvar1 = zvar1
            self.cost1 = cost1            
                
            # Also store the estimates as zhat and zhatvar to avoid the
            # confusing names, z1, zvar1
            self.zhat = z1
            self.zhatvar = zvar1
            
            # Msg passing to est 2 (from est 1 =>idir=0)
            r2, rvar2 = self.msg_hdl.msg_sub(z1,zvar1,idir=0)
                                         
            # Compute total cost
            if self.comp_cost:
                var_cost, self.grad, self.con = self.msg_hdl.get_cost_terms()
                self.var_cost1, self.var_cost2 = var_cost            
                self.cost = self.cost1 + self.cost2 - self.var_cost1 \
                    - self.var_cost2 + self.msg_hdl.Hgauss(self.zvar1)
            t2 = time.time()
            self.time_est1 = t2-t1
            self.time_iter = t2-t0
            
            # Print progress
            if self.prt_period > 0:
                if (it % self.prt_period == 0):
                    if self.comp_cost:
                        print("it={0:4d} cost={1:12.4e} con={2:12.4e} grad={3:12.4e}".format(\
                            it, self.cost, self.con, self.grad))
                    else:
                        print("it={0:4d}".format(it))

            
            # Save history
            self.save_hist()
            self.r2 = r2
            self.rvar2 = rvar2
        
              
    
            
            

