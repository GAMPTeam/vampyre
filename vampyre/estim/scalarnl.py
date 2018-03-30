"""
scalarnl.py:  Estimation methods for scalar nonlinear functions
"""
from __future__ import division

import numpy as np

# Import other subpackages in vampyre
import vampyre.common as common

# Import individual classes and methods from the current sub-package
from vampyre.estim.base import BaseEst

class ScalarNLEst(BaseEst):
    """
    Base class for an esitmator for a general scalar nonlinear penalty
    
    This esitmator corresponds to a general nonlinear penalty :math:`f(z)`.
    The penalty function is defined in the derived class by implementing
    the method :code:`fnl`.  Right now, the function only implements
    MAP estimation which it performs via Newton's method.
    
    :param shape:  Shape of :math:`z`.
    :param var_axes:  The axes on which the input variance is repeated.
    :param step_init:  Intial gradient descent step-size
    :param step_max:  Max gradient descent step-size
    :param step_min:  Min gradient descent step-size
    :param max_it:  Max number of gradient-descent iterations per call to the 
        estimation method
    :param zinit:  Initial estimate for :math:`z`
    :param is_complex:  Flag indicating if :math:`z` is complex.
    :param gtol:  gradient norm tolerance.
        
    """    
    def __init__(self,shape,var_axes=(0,),max_it=10, step_init=1,\
        dtype=np.float64,name=None,\
        step_max=1,step_min=1e-8,zinit=None,is_complex=False,gtol=1e-3):
        
        BaseEst.__init__(self,shape=shape,var_axes=var_axes,dtype=dtype,\
            name=name, type_name='ScalarNL', nvars=1, cost_avail=True)

        # Save parameters
        self.max_it = max_it
        self.cost_avail = True
        self.step_last = step_init
        self.step_max = step_max
        self.step_min = step_min
        self.is_complex = is_complex
        self.gtol = gtol
        self.min_it = 1
        
        # Set default value that the Hessian is available
        self.hess_avail = True
                
        # Set initial point
        if (zinit is None):
            self.zlast = self.proj(np.zeros(self.shape))
        else:
            self.zlast = zinit
            
    def proj(self,z):
        """
        Projects any value :math:`z` onto the feasible set.  
        The default implementation performs no projection
        """
        return z
        
    def fnl(self,z):
        """
        Penalty function.  This must be implemented in the derived class.
        
        :param z:  The input to the penalty function.
        
        :returns:  :code:`f,fgrad,[fhess]` The function, its gradient and
           second derivative.  The second derivative must be implemented
           only in the case where :code:`self.hess_avail == True`.
        """
        raise NotImplementedError()
            
    def fnl_aug(self,z,r,rvar):
        """
        Augmented nonlinear function and its gradient.
        
        Given the penalty function :math:`f(z)`, the function returns
        the value and gradient of the augmented function,
        
        :math:`f_{aug}(z,r) = f(z) + (1/\\tau_r)|r-z|^2            
        """
        
        # Evaluate function
        if self.hess_avail:        
            f0, fgrad0, fhess0 = self.fnl(z)
        else:
            f0, fgrad0 = self.fnl(z)
                    
        # Add the augmenting term
        rvar_rep = common.repeat_axes(rvar,self.shape,self.var_axes,rep=False)
        aug = np.abs(z-r)**2/rvar_rep
        aug_grad = 2*np.conj(z-r)/rvar_rep
        aug_hess = 2/rvar_rep
        if not self.is_complex:
            aug /= 2
            aug_grad /= 2
            aug_hess /= 2
        faug = np.sum(f0 + aug)
        faug_grad = fgrad0 + aug_grad
        if self.hess_avail:
            faug_hess = fhess0 + aug_hess                         
            return faug, faug_grad, faug_hess
        else:
            return faug, faug_grad            
        
    def est_init(self, return_cost=False,ind_out=None,\
        avg_var_cost=True):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.  
        
        The default implementation calls :code:`est` with the initial variance
        (which is typically large)
        
        :param boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """        
        return self.est(self.rinit,self.rvar_init,return_cost,ind_out,\
            avg_var_cost)

    def est(self,r,rvar,return_cost=False,ind_out=None,\
        avg_var_cost=True):
        """
        Estimation function
        
        The proximal estimation function as 
        described in the base class :class:`vampyre.estim.base.Estim`
                
        :param r: Proximal mean
        :param rvar: Proximal variance
        :param Boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        
        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior
            mean, variance and optional cost.
        """        
        
        # Check parameters
        if (ind_out != [0]) and (ind_out != None):
            raise ValueError("ind_out must be either [0] or None")
        if not avg_var_cost:
            raise ValueError("disabling variance averaging not supported for MixEst")

        # Check that the Hessian is available
        if not self.hess_avail:
            raise Exception("Second derivative must be currently supported.  "+\
                    "This is needed for the variance computation in the "+\
                    "current implementation.")
        
        
        # Get point from previous run
        z = self.zlast
        step = self.step_last
        
        # Compute initial function and step direction
        if self.hess_avail:
            f, fgrad, fhess = self.fnl_aug(z,r,rvar)
            g = fgrad / fhess
            zhatvar = 1 / fhess
        else:
            f, fgrad = self.fnl_aug(z,r,rvar)
            g = fgrad
            
        a = 0.5
        done = False
        it = 0
        while not done:        
            # Try test point
            z1 = z - step*g
            z1 = self.proj(z1)
            if self.hess_avail:
                f1, fgrad1, fhess1 = self.fnl_aug(z1,r,rvar)                
                g1 = fgrad1/fhess1
            else:
                f1, fgrad1 = self.fnl_aug(z1,r,rvar)
                g1 = fgrad1
            
            
            # Compute expected decrease
            dfest = np.sum(np.conj(fgrad)*(z1-z))
            
            # Accept or reject point
            if (f1-f < a*dfest) and (dfest < 0):
                z = z1
                f = f1
                fgrad = fgrad1
                g = g1
                step = 2*step
                step = np.minimum(self.step_max, step)
                zhatvar = 1/fhess1
                
            else:
                step = 0.5*step
                step = np.maximum(self.step_min, step)
                
            # Check termination
            it += 1
            gnorm = np.mean(np.abs(g)**2)
            done = (it >= self.max_it) or (gnorm < self.gtol)
            done = done and (it >= self.min_it)
                            
        # Save results
        self.zlast = z
        self.step_last = step
        self.nit_last = it
                
        # Average the variance
        zhatvar = np.mean(zhatvar, axis=self.var_axes)
        if self.is_complex:
            zhatvar /= 2
                    
        if return_cost:
            return z, zhatvar, f
        else:
            return z, zhatvar
            
class LogisticEst(ScalarNLEst):
    def __init__(self,y,var_axes=(0,),max_it=100,gtol=1e-6,\
        rinit=None,rvar_init=10,name=None,dtype=np.float64):
        """
        Lotistic estimator with binary class label.
        
        The penalty function is given by,
        
        :math:`f(z,y) = z - zy + \\log(1 + \\exp(-z))`
        
        To avoid overflow, this is implemented as
        
        :math:`f(z,y) = \\max(z,0) - zy + \\log(1 + \\exp(-|z|))`
                    
        :param y:  Binary class labels 0 or 1.
        :param nit_max:  Maximum number of Newton iterations
        :param gtol:  Stopping tolerance for optimization
        :param rinit:  Initial prior mean on z for :code:`est_init`
        :param rvar_init:  Initial prior variance on z for :code:`est_init`
        
        :note:  The penalty matches the Tensorflow operator
           :code:`tf.nn.sigmoid_cross_entropy_with_logits`
        """
        # Save parameters
        shape = y.shape
        if rinit is None:
            self.rinit = np.zeros(shape)
        else:
            self.rinit = rinit
        
        # Intialize the base class              
        #ScalarNLEst.__init__(self,shape,var_axes,max_it=max_it,\
        #    step_init=1, step_max=1,step_min=1e-8,zinit=rinit)
        ScalarNLEst.__init__(self,shape,var_axes=var_axes,max_it=max_it,\
            step_init=1,dtype=dtype,name=name,\
            step_max=1,step_min=1e-8,zinit=self.rinit)
            
        if np.isscalar(rvar_init):
            var_shape = common.get_var_shape(self.shape,self.var_axes)
            rvar_init = np.tile(rvar_init, var_shape)
        self.rvar_init = rvar_init                        
        
        # Indicate that the Hessian is available
        self.hess_avail = True
        
        # Save class label
        self.y = y

    def fnl(self,z):
        """
        Logistic function
        
        :returns:  :code:`f,fgrad,fhess` the logistic function, its gradient
           and hessian
        """
        p = 1/(1+np.exp(-z))
        q = 1/(1+np.exp(z))
        f = np.maximum(z,0) - self.y*z + np.log(1+np.exp(-np.abs(z)))
        fgrad = p-self.y 
        fhess = p*q        
        return f, fgrad, fhess            
            
        
