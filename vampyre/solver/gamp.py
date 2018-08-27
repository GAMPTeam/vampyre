"""
gamp.py:  Main GAMP solver
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


class Gamp(Solver):
    """
    Gamp solver
    
    The GAMP solver performs estimation on a penalty function of the form:
    
    :code:`f(z) = f0(z0) + f1(z1) s.t. z1=A.dot(z0)`
    
    where :math:`f0(z)` and :math:`f1(z)` are two penalty functions
    
    :param est0:  Estimator corresonding to :math:`f0(z)`
    :param est1:  Estimator corresonding to :math:`f1(z)`
    :param A:  Linear transform
    :param hist_list:  List of attributes to save the history of during the
       course of the iterations.
    :param nit:  Maximum number of iterations          
    :param comp_cost:  Compute cost
    :param prt_period:  Period for printing progress (value of 0 indicates
        no printing)
    :param step_adapt:  Enable adaptive step size
    :param step:  Initial step size
    """
    def __init__(self, est0, est1, A, hist_list=[], nit=10,\
        comp_cost=True, prt_period=0, is_complex=False, map_est=False,\
        step_adapt=False,step=0.95,step_min=1e-6):
        Solver.__init__(self,hist_list)
        self.est0 = est0
        self.est1 = est1
        self.A = A        
        self.nit = nit
        self.comp_cost = comp_cost
        self.prt_period = prt_period
        self.is_complex = is_complex
        self.map_est = map_est

        # Get dimensions
        self.shape0 = A.shape0
        self.shape1 = A.shape1
        self.var_axes0 = A.var_axes0
        self.var_axes1 = A.var_axes1
        
        # Step size parameters
        self.step = step
        self.step_adapt = step_adapt   
        self.step_min = step_min    # min step size
        self.step_dec = 0.5     # decrement on a failure
        self.step_inc = 1.1     # decrement on a failure
        
                        
        # Check if dimensions match
        if self.est0.shape != self.shape0:
            err_str = '%s shape %s does not match operator input shape %s' %\
                (self.est0.name, str(self.est0.shape), str(self.shape0))                
            raise ValueError(err_str)
        if self.est1.shape != self.shape1:
            err_str = '%s shape %s does not match operator output shape %s' %\
                (self.est1.name, str(self.est1.shape), str(self.shape1))                
            raise ValueError(err_str)
        if self.est0.var_axes != self.var_axes0:
            err_str = '%s var_axes %s does not match operator input var_axes %s' %\
                (self.est0.name, str(self.est0.var_axes), str(self.var_axes0))                
            raise ValueError(err_str)
        if self.est1.var_axes != self.var_axes1:
            err_str = '%s var_axes %s does not match operator input var_axes %s' %\
                (self.est1.name, str(self.est1.var_axes), str(self.var_axes1))                
            raise ValueError(err_str)
                            
    def summary(self):
        """
        Prints a summary of the model
        """
        print('est0: %s (%s) shape: %s' % (str(self.est0.name),\
                         str(self.est0.type_name),str(self.shape0)))
        print('est1: %s (%s) shape: %s' % (str(self.est1.name),\
                         str(self.est1.type_name),str(self.shape1)))
        
                        
    def solve(self):
        """
        Runs the main GAMP algorithm
        
        The final estimates are saved in :code:`z0` and :code:`z1`
        along with variances :code:`zvar0` and :code:`zvar1`
        """
                    
        # Check if cost is available for both estimators
        if not self.est0.cost_avail or not self.est1.cost_avail:
            self.comp_cost = False
            
        # Initial esitmate from the input node
        if self.comp_cost:
            z0, zvar0, cost0 = self.est0.est_init(return_cost=True)
        else:
            z0, zvar0 = self.est0.est_init(return_cost=False)
            cost0 = 0
        self.z0 = z0
        self.zvar0 = zvar0
        self.cost0 = cost0
                
        # Initialize other variables
        self.var_cost0 = 0
        self.var_cost1 = 0
        self.cost = 0
        self.s = np.zeros(self.shape1)
                
        for it in range(self.nit):
            
            # Forward transform to est1
            t0 = time.time()
            rvar1_new = self.A.var_dot(self.zvar0)
            rvar1_rep = common.repeat_axes(rvar1_new,self.shape1,\
                                              self.var_axes1,rep=False)
            z1_mult = self.A.dot(self.z0)
            r1_new = z1_mult - rvar1_rep*self.s
            
            # Damping
            if it > 0:                
                self.r1 = (1-self.step)*self.r1 + self.step*r1_new
                self.rvar1 = (1-self.step)*self.rvar1 + self.step*rvar1_new
            else:
                self.r1 = r1_new
                self.rvar1 = rvar1_new

            # Estimator 1            
            if self.comp_cost:  
                z1, zvar1, cost1 = self.est1.est(self.r1, self.rvar1, return_cost=True)               
                if not self.map_est:
                    cost1 -= self.cost_adjust(self.r1,z1,self.rvar1,zvar1,\
                                                   self.shape1,self.var_axes1)
            else:
                z1, zvar1 = self.est1.est(self.r1, self.rvar1, return_cost=False)               
                cost1 = 0
            self.z1 = z1
            self.zvar1 = zvar1
            self.cost1 = cost1    
            con_new = np.mean(np.abs(z1-z1_mult)**2)                    
            
            # Reverse nonlinear transform to est 0
            self.s = (self.z1-self.r1)/rvar1_rep
            self.sprec = 1/self.rvar1*(1-self.zvar1/self.rvar1)
            t1 = time.time()
            self.time_est1 = t1-t0
                                    
            # Reverse linear transform to est 0 
            rvar0_new = 1/self.A.var_dotH(self.sprec)
            rvar0_rep = common.repeat_axes(rvar0_new,self.shape0,\
                                              self.var_axes0,rep=False)
            r0_new = self.z0 + rvar0_rep*self.A.dotH(self.s)
            
            # Damping
            if it > 0:
                self.r0 = (1-self.step)*self.r0 + self.step*r0_new
                self.rvar0 = (1-self.step)*self.rvar0 + self.step*rvar0_new
            else:
                self.r0 = r0_new
                self.rvar0 = rvar0_new
                
                    
            # Estimator 0
            if self.comp_cost:
                z0, zvar0, cost0 = self.est0.est(self.r0, self.rvar0, return_cost=True)
                if not self.map_est:
                    cost0 -= self.cost_adjust(self.r0,z0,self.rvar0,zvar0,\
                                                   self.shape0,self.var_axes0)
                
            else:
                z0, zvar0 = self.est0.est(self.r0, self.rvar0, return_cost=False)
                cost0 = 0
            self.z0 = z0
            self.zvar0 = zvar0
            self.cost0 = cost0           

                                                                    
            # Compute total cost and constraint          
            cost_new = self.cost0 + self.cost1 
            if not self.map_est:
                cost_new += self.cost_gauss()
                
            # Step size adaptation
            if (self.step_adapt) and (it > 0):
                if (con_new < self.con):
                    self.step = np.minimum(1,self.step_inc*self.step)
                else:
                    self.step = np.maximum(self.step_min, self.step_dec*self.step)
            self.cost=cost_new
            self.con=con_new
            
            t2 = time.time()
            self.time_est0 = t2-t1
            self.time_iter = t2-t0
            
            # Print progress
            if self.prt_period > 0:
                if (it % self.prt_period == 0):
                    if self.comp_cost:
                        print("it={0:4d} cost={1:12.4e} con={2:12.4e} step={3:12.4e}".format(\
                            it, self.cost, self.con, self.step))
                    else:
                        print("it={0:4d} con={1:12.4e}".format(\
                            it, self.con))
          
            # Save history
            self.save_hist()
              
    
    def cost_adjust(self,r,z,rvar,zvar,shape,var_axes):
        """
        Computes the cost adjustment term for the
        Bethe Free Energy:
            
            J = beta*[log(2*pi*rvar) + ((z-r)**2 + xvar)/rvar]
            
        where beta = 1 for complex problems and 0 for real problems
        """    
        J0 = np.mean(np.log(2*np.pi*rvar))*np.product(shape)
        rvar_rep = common.repeat_axes(rvar,shape,\
                                      var_axes,rep=False)
        J1 = np.sum(np.abs(r-z)**2/rvar_rep)
        J2 = np.mean(zvar/rvar)*np.product(shape)
        J = J0 + J1 + J2
        if not self.is_complex:
            J = J / 2
        return J
    
    def cost_gauss(self):
        """
        Compute the Gaussian term in the BFE
        """
        J = np.mean(self.zvar1/self.rvar1 + np.log(2*np.pi*self.rvar1))
        J = J*np.product(self.shape1)
        if not self.is_complex:
            J = J / 2
        return J
        
            

