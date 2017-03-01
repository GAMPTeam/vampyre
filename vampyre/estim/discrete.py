import numpy as np

# Import other sub-packages
import vampyre.common as common

# Import from other modules in the same package
from vampyre.estim.base import Estim

class DiscreteEst(Estim):
    """ Discrete estimator class
    
    An estimator corresponding to a discrete density with scalar values.
    
    :param zval: vector of values for each component of the array :code:`z`.  
        These are a list of scalars.
    :param pz: vector of probabilities of each value
    :param shape:  shape of the unknown array.
    :param var_axes: axes on which the variance is to be averaged.
       Default is 'all'
       
    :note:  The class only currently supports MMSE estimation
    """    
    def __init__(self, zval, pz, shape, var_axes=(0,),\
                 is_complex=False):
        Estim.__init__(self)
        
        # Convert scalars to arrays
        if np.isscalar(zval):
            zval = np.array([zval])
        if np.isscalar(pz):
            pz = np.array([pz])
            
        # Set parameters
        self.zval = zval
        self.pz = pz
        self.shape = shape
        self.is_complex = is_complex
        self.fz = -np.log(pz)
                
        # Set the variance axes
        if var_axes == 'all':
            ndim = len(shape)
            var_axes = tuple(range(ndim))
        self.var_axes = var_axes        
        self.cost_avail = True
                                 
    def est_init(self, return_cost=False, avg_var_cost=True):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.
                
        :param Boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :param Boolean avg_var_cost: Average variance and cost.
            This should be disabled to obtain per element values.
            (Default=True)            
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """     

        # Compute the scalar mean, variance and cost          
        zmean = np.sum(self.pz*self.zval)
        zvar = np.sum(self.pz*(self.zval-zmean)**2)
        cost = 0
        
        # Repeat the mean value to all axes
        zmean = np.tile(zmean, self.shape)
        
        # Repeat the variance to all axes that are not averaged over
        ndim = len(self.shape)
        axes_spec = [i for i in range(ndim) if i not in self.var_axes]
        if axes_spec != []:
            shapea = np.array(self.shape)
            zvar = np.tile(zvar, shapea[axes_spec])

        if not avg_var_cost:
            cost = np.tile(cost,self.shape)            
            
        if return_cost:
            return zmean, zvar, cost
        else:
            return zmean, zvar                                
                    
    def est(self,r,rvar,return_cost=False,avg_var_cost=True):
        """
        Estimation function
        
        The proximal estimation function as 
        described in the base class :class:`vampyre.estim.base.Estim`
                
        :param r: Proximal mean
        :param rvar: Proximal variance
        :param boolean return_cost:  Flag indicating if :code:`cost` 
            is to be returned
        :param Boolean avg_var_cost: Average variance and cost.
            This should be disabled to obtain per element values.
            (Default=True)            
        
        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior 
        mean, variance and optional cost.
        """
        # Infinite variance case
        if np.any(rvar==np.Inf):
            return self.est_init(return_cost, avg_var_cost)
     
        
        # Convert to 1D vectors        
        r1 = r.ravel()
        rvar1 = common.repeat_axes(rvar,self.shape,self.var_axes)
        rvar1 = rvar1.ravel()
        
        # Compute the augmented penalty for each value
        faug = (np.abs(self.zval[None,:]-r1[:,None])**2)/rvar1[:,None]
        if not self.is_complex:
            faug *= 0.5
        faug = faug + self.fz[None,:]

        # Compute the conditional probability of each value        
        fmin = np.min(faug,axis=1)
        pzr = np.exp(-faug + fmin[:,None])
        psum = np.sum(pzr,axis=1)
        pzr = pzr / psum[:,None]
        cost = -np.log(psum) + fmin
        
        zhat = pzr.dot(self.zval)
        zerr = np.abs(self.zval[None,:]-zhat[:,None])**2
        zhatvar = np.sum(pzr*zerr,axis=1)
        
        # Reshape values
        cost = np.reshape(cost, self.shape)
        zhat = np.reshape(zhat, self.shape)
        zhatvar = np.reshape(zhatvar, self.shape)
        
        self.pzr = pzr
        
        # Average values
        if avg_var_cost:
            cost = np.sum(cost)
            zhatvar = np.mean(zhatvar,axis=self.var_axes)
        
        if return_cost:
            return zhat, zhatvar, cost
        else:
            return zhat, zhatvar
                