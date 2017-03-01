"""
vamp.py:  Main VAMP solver
"""
from __future__ import division

# Import general packages
import numpy as np
import time

# Import other subpackages
import vampyre.estim as estim
import vampyre.trans as trans
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
    """
    def __init__(self, est1, est2, msg_hdl=[], hist_list=[], nit=10,\
        comp_cost=False):
        Solver.__init__(self,hist_list)
        self.est1 = est1
        self.est2 = est2
        self.msg_hdl = msg_hdl        
        self.nit = nit
        self.comp_cost = comp_cost
                        
    def solve(self):
        """
        Runs the main VAMP algorithm
        
        The final estimates are saved in :code:`r1` and :code:`r2`
        along with variances :code:`rvar1` and :code:`rvar2`
        """
                    
        # Check if cost is available for both estimators
        if not self.est1.cost_avail or not self.est2.cost_avail or self.msg_hdl == []:
            self.comp_cost = False
            
        # Set to default variance handler if not specified
        if self.msg_hdl == []:
            self.msg_hdl = estim.MsgHdlSimp()

        # Initial esitmate
        if self.comp_cost:
            r2, rvar2, cost1 = self.est1.est_init(return_cost=True)
        else:
            r2, rvar2 = self.est1.est_init(return_cost=False) 
            cost1 = 0
        self.r2 = r2
        self.rvar2 = rvar2
        self.cost1 = cost1
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
            
            # Variance cost
            if self.comp_cost:
                self.var_cost2 = self.msg_hdl.cost(z2,zvar2,r2,rvar2)
            
            # Msg passing to est 1
            r1, rvar1 = self.msg_hdl.msg_sub(z2,zvar2,r2,rvar2,self.r1,self.rvar1)
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
            if self.comp_cost:
                self.var_cost1 = self.msg_hdl.cost(z1,zvar1,r2,rvar1)
                
            # Also store the estimates as zhat and zhatvar to avoid the
            # confusing names, z1, zvar1
            self.zhat = z1
            self.zhatvar = zvar1
            
            # Msg passing to est 2
            r2, rvar2 = self.msg_hdl.msg_sub(z1,zvar1,r1,rvar1,self.r2,self.rvar2)
                                
            # Compute total cost
            if self.comp_cost:
                self.cost = self.cost1 + self.cost2 - self.var_cost1 \
                    - self.var_cost2 + self.msg_hdl.Hgauss(self.zvar1)
            t2 = time.time()
            self.time_est1 = t2-t1
            
            # Save history
            self.save_hist()
            self.r2 = r2
            self.rvar2 = rvar2
        