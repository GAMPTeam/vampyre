"""
relu.py:  Estimator and test for the rectified linear unit
"""
import numpy as np
import scipy.special
from scipy.integrate import quad

# Import other subpackages in vampyre
import vampyre.common as common

# Import individual classes and methods from the current subpackage
from vampyre.estim.base import Estim
from vampyre.estim.interval import gauss_integral
       
class ReLUEstim(Estim):    
    """
    Estimatar for a rectified linear unit
    
    :math:`z_1 = f(z_0) = \\max(0,z_0)`
    
    :param shape:  shape of :math:`z_0` and :math:`z_1`
    :param z0rep_axes:  axes on which the variance for :math:`z_0` is averaged
    :param z1rep_axes:  axes on which the variance for :math:`z_1` is averaged
    """        
    def __init__(self,shape,z0rep_axes=(0,), z1rep_axes=(0,)):
        Estim.__init__(self)
        self.shape = shape
        ndim = len(shape)
        if z0rep_axes == 'all':
            z0rep_axes = tuple(range(ndim))
        if z1rep_axes == 'all':
            z1rep_axes = tuple(range(ndim))            
        self.z0rep_axes = z0rep_axes
        self.z1rep_axes = z1rep_axes
        self.cost_avail = True
        
        # Initial variances
        self.zvar0_init= np.Inf
        self.zvar1_init= np.Inf
    
    def est_init(self,return_cost=False):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.
                
        :param Boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """          
        zhat0   = np.zeros(self.shape)
        zhat1   = np.zeros(self.shape)
        zhat    = [zhat0,zhat1]
     
        # Compute the shapes for the variance and set the initial value of
        # the variance according to the shape
        ndim = len(self.shape)
        axes_spec = [i for i in range(ndim) if i not in self.z0rep_axes]
        if axes_spec == []:
            zvar0 = self.zvar0_init
        else:
            shape1 = tuple(np.array(self.shape)[axes_spec])
            zvar0 = np.tile(self.zvar0_init, shape1)
        axes_spec = [i for i in range(ndim) if i not in self.z1rep_axes]
        if axes_spec == []:
            zvar1 = self.zvar1_init
        else:
            shape1 = tuple(np.array(self.shape)[axes_spec])
            zvar1 = np.tile(self.zvar1_init, shape1)            
        zvar = [zvar0,zvar1]

        cost = 0
        if return_cost:
            return zhat, zvar, cost
        else:
            return zhat, zvar            
    

    def est(self,r,rvar,return_cost=False):
        """
        Estimation function
        
        The proximal estimation function as 
        described in the base class :class:`vampyre.estim.base.Estim`
                
        :param r: Proximal mean
        :param rvar: Proximal variance
        :param boolean return_cost:  Flag indicating if :code:`cost` 
            is to be returned
         
        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior 
        mean, variance and optional cost.
        """
        
        """
        In the MMSE estimation case, we wish to estimate
        z0 and z1 with priors zi = N(ri,rvari) and z1=f(z0)
        
        Substituting in z1 = f(z0), we have the density of z0:
          
           p(z0)  \propto qn(z0)1_{z0 < 0}  + qp(z0)1_{z0 > 0}
           
        where
           
           qp(z0)  = exp[-(z0-r0)^2/(2*rvar0) - (z0-r1)^2/(2*rvar1)]
           qn(z0)  = exp[-(z0-r0)^2/(2*rvar0) - r1^2/(2*rvar1)]
           
        First, we complete the squares and write:
        
           qp(z0) = exp(Amax)*Cp*exp(-(z0-rp)^2/(2*zvarp))/sqrt(2*pi*zvarp)  
           qn(z0) = exp(Amax)*Cn*exp(-(z0-rn)^2/(2*zvarn))/sqrt(2*pi*zvarn)        
           
        """
                 
        # Unpack the terms
        r0, r1 = r
        rvar0, rvar1 = rvar
        
        # Reshape the variances
        rvar0 = common.repeat_axes(rvar0,self.shape,self.z0rep_axes)
        rvar1 = common.repeat_axes(rvar1,self.shape,self.z1rep_axes)
                 
        # Compute the conditional Gaussian terms for z > 0 and z < 0
        zvarp = rvar0*rvar1/(rvar0+rvar1)
        zvarn = rvar0
        rp = (rvar1*r0 + rvar0*r1)/(rvar0+rvar1)        
        rn = r0

        # Compute scaling constants for each region
        Ap = 0.5*((rp**2)/zvarp - (r0**2)/rvar0 - (r1**2)/rvar1)
        An = 0.5*(-(r1**2)/rvar1 + np.log(zvarn))
        Amax = np.maximum(Ap,An)
        Ap = Ap - Amax
        An = An - Amax
        Cp = np.exp(Ap)
        Cn = np.exp(An)
                        
        # Compute moments for each region
        zp = Cp*gauss_integral(0, np.Inf, rp, zvarp)
        zn = Cn*gauss_integral(-np.Inf, 0, rn, zvarn)
        
        # Compute mean        
        zhat0 = (zp[1] + zn[1])/(zp[0] + zn[0])
        zhat1 = zp[1]/(zp[0] + zn[0])
        zhat = [zhat0,zhat1]     
        
        # Compute the variance
        zhatvar0 = (zp[2] + zn[2])/(zp[0]+zn[0]) - zhat0**2
        zhatvar1 = zp[2]/(zp[0]+zn[0]) - zhat1**2
        
        # Average the variance over the specified axes
        zhatvar0 = np.mean(zhatvar0,axis=self.z0rep_axes)
        zhatvar1 = np.mean(zhatvar1,axis=self.z1rep_axes)
        zhatvar = [zhatvar0,zhatvar1]

        if not return_cost:        
            return zhat, zhatvar
            
        """
        Compute the 
            cost = -\log \int p(z_0) 
                 = -Amax - log(zp[0] + zn[0])        
        """
        nz = np.prod(self.z0rep_axes)
        cost = -nz*(Amax - np.mean(np.log(zp[0] + zn[0])))
        return zhat, zhatvar, cost

def relu_test(zshape=(1000,10),tol=0.15,verbose=False):
    """
    ReLUEstim unit test.
    
    In this test, a matrix :math:`z_0=r_0+w_0` is generated with iid Gaussian 
    components, :math:z_1 = \\max(0,z_0)` is the ReLU ouptut of :math:`z_0`
    and :math:`r_1 = z_1 + w_1`.  The ReLU estimator is then used to
    estimate :math:`z_i` from :math:`r_i`.  The test passes if the 
    predicted variance matches the true variance with a specified tolerance.
            
    :param zshape: shape of :math:`z_0` and :math:`z_1`.
    :param tol:  Tolerance for test test to pass
    :param Boolean verbose:  prints results.  
    """
     
    # Get number of columns
    ns = zshape[1]
        
    # Set random parameters
    rvar0 = np.power(10,np.random.uniform(-2,1,ns))
    rvar1 = np.power(10,np.random.uniform(-2,1,ns))
        
    # Construct random input
    r0 = np.random.normal(0,1,zshape) 
    z0 = r0 + np.random.normal(0,1,zshape)*np.sqrt(rvar0)[None,:]
    
    # Perform ReLU and add noise
    z1 = np.maximum(z0,0)
    r1 = z1 + np.random.normal(0,1,zshape)*np.sqrt(rvar1)[None,:]
    
    # Construct the estimator
    relu = ReLUEstim(shape=zshape)
    
    # Repeath the variances
    r = [r0,r1]
    rvar = [rvar0,rvar1]
    zhat, zhatvar, cost = relu.est(r,rvar,return_cost=True)
    
    # Unpack the estimates
    zhat0, zhat1 = zhat
    zhatvar0, zhatvar1 = zhatvar
    
    # Compute the true error and compare to the 
    zerr0 = np.mean((zhat0-z0)**2,axis=0)
    zerr1 = np.mean((zhat1-z1)**2,axis=0)
    
    # Compute average difference
    diff0 = np.mean(np.maximum(zhatvar0/zerr0,zerr0/zhatvar0))-1
    diff1 = np.mean(np.maximum(zhatvar1/zerr1,zerr1/zhatvar1))-1
    
    # Check if fails
    fail = (diff0 > tol) or (diff1 > tol)    
    if verbose or fail:
        print("")
        print("z0 err: act: {0:12.4e} pred: {1:12.4e} diff: {2:12.4e}".format(\
            np.mean(zerr0),np.mean(zhatvar0),diff0))
        print("z1 err: act: {0:12.4e} pred: {1:12.4e} diff: {2:12.4e}".format(\
            np.mean(zerr1),np.mean(zhatvar1),diff1)) 
    if fail:
        raise common.TestException("Posterior predicted variance does not match "+\
            "actual variance within tolerance")
    