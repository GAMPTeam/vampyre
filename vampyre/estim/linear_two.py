"""
linear_two.py:  Estimation for general linear constraints with Gaussian noise.
"""
from __future__ import division

import numpy as np
import scipy

# Import other subpackages in vampyre
import vampyre.common as common
import vampyre.trans as trans

# Import individual classes and methods from the current sub-package
from vampyre.estim.base import Estim

class LinEstimTwo(Estim):
    """
    Esitmator based on a linear constraint with noise
    
    This esitmator corresponds to a linear constraint of the form
    :math:`z_1 = Az_0 + b + w` 
    where :math:`w \\sim {\\mathcal N}(0,\\tau_w I)`.
    Specifically, the penalty is 
    
    :math:`f(z) = (1/2\\tau_w)\|z_1-Az_0-b\|^2 + (d/2)\\ln(2\\pi \\tau_w)`,
    
    where :math:`d` is the dimension of :math:`b`
    
    :param A: Linear operator represented as a 
        :class:`vampyre.trans.base.LinTrans` 
    :param b: Constant term
    :param wvar:  Noise level        
    :param wrep_axes':  The axes on which the output noise variance is repeated.
        Default is 'all'.  
    :param zrep_axes:  The axes on which the input variance is repeated.
        Default is 'all'.
    :param Boolean is_complex:  indiates if :math:`z` is complex    
    :param Boolean map_est:  indicates if estimator is to perform MAP 
        or MMSE estimation. This is used for the cost computation.  
    :param est_meth: Estimation method.  `svd` or `cg` corresponding to
        an SVD-based method or conjugate gradient.
    :param nit_cg:  Maximum number of CG iterations.  Note the CG optimization
        is warm-started with the previous value.
       
    :note:  The linear operator :code:`A` must have :code:`svd_avail==True`
        to use `est_meth = svd`.  Otherwise, use `est_meth=cg`.        
    :note:  The linear operator must also have the :code:`shape0` and
       :code:`shape1` arrays available to compute the dimensions.
       
    :note:  The axes :code:`wrep_axes` and :code:`zerp_axes` must 
       include the axis in which :code:`A` operates for the SVD method.
    """    
    def __init__(self,A,b,wvar=0,\
                 z1rep_axes=(0,), z0rep_axes=(0,),wrep_axes='all',\
                 map_est=False,is_complex=False,est_meth='svd',nit_cg=100):
        
        Estim.__init__(self)
        self.A = A
        self.b = b
        self.wvar = wvar
        self.map_est = map_est
        self.is_complex = is_complex
        self.cost_avail = True
        
        # Initial variance.  This is large value since the quantities
        # are underdetermined
        self.zvar0_init = np.Inf
        self.zvar1_init = np.Inf
        
        # Get the input and output shape
        self.shape0 = A.shape0        
        self.shape1 = A.shape1
        
        # Set the repetition axes
        ndim = len(self.shape1)
        if z0rep_axes == 'all':
            z0rep_axes = tuple(range(ndim))        
        if z1rep_axes == 'all':
            z1rep_axes = tuple(range(ndim))
        if wrep_axes == 'all':
            wrep_axes = tuple(range(ndim))            
        self.z0rep_axes = z0rep_axes
        self.z1rep_axes = z1rep_axes        
        self.wrep_axes = wrep_axes        
        
        # Initialization depending on the estimation method
        self.est_meth = est_meth
        if self.est_meth == 'svd':
            self.init_svd()
        elif self.est_meth == 'cg':
            self.init_cg()
        else:
            raise common.VpException(
                "Unknown estimation method {0:s}".format(est_meth))
        
        # CG parameters
        self.nit_cg = nit_cg
                
    def init_cg(self):
        """
        Initialization that is specific to the conjugate gradient method
        """
        pass
       
                    
    def init_svd(self):
        """
        Initialization for the SVD method 
        """
        # Compute the SVD terms
        # Take an SVD A=USV'.  Then write p = SV'z + w,
        if not self.A.svd_avail:
            raise common.VpException("Transform must support an SVD")
        self.bt = self.A.UsvdH(self.b)
        srep_axes = self.A.srep_axes
        
        # Compute the norm of ||b-UU*(b)||^2/wvar
        if np.all(self.wvar > 0):
            bp = self.A.Usvd(self.bt)
            wvar_rep = common.repeat_axes(self.wvar, self.shape1, self.wrep_axes, rep=False)
            err = np.abs(self.b-bp)**2
            self.bpnorm = np.sum(err/wvar_rep)
        else:
            self.bpnorm = 0
                        
        # Check that all axes on which A operates are repeated 
        ndim = len(self.shape1)
        for i in range(ndim):
            if not (i in self.z1rep_axes) and not (i in srep_axes):                
                raise common.VpException(
                    "Variance must be constant over output axis")
            if not (i in self.wrep_axes) and not (i in srep_axes):                
                raise common.VpException(
                    "Noise variance must be constant over output axis")                    
            if not (i in self.z0rep_axes) and not (i in srep_axes):                
                raise common.VpException(
                    "Variance must be constant over input axis")
        
                            
        
    def est_init(self, return_cost=False):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.
        
        Since the system is underdetermined, the initial estimator
        just produces values of the correct shape.
        
        :param boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """       
        
        zmean0 = np.zeros(self.shape0)
        zmean1 = np.zeros(self.shape1)
        zmean = [zmean0,zmean1]
        
        # Compute the shapes for the variance
        zvar0_shape = np.mean(zmean0, self.z0rep_axes).shape
        zvar0 = np.tile(self.zvar0_init, zvar0_shape)
        zvar1_shape = np.mean(zmean1, self.z1rep_axes).shape
        zvar1 = np.tile(self.zvar1_init, zvar1_shape)
        zvar = [zvar0,zvar1]
        
        if return_cost:
            cost = 0
            return zmean, zvar, cost
        else:
            return zmean, zvar
                    
    def est(self,r,rvar,return_cost=False):
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
        if self.est_meth == 'svd':
            return self.est_svd(r,rvar,return_cost)
        elif self.est_meth == 'cg':
            return self.est_cg(r,rvar,return_cost)
        else:
            raise common.VpException(
                "Unknown estimation method {0:s}".format(self.est_meth))
                
    def est_cg(self,r,rvar,return_cost=False):        
        """
        CG-based estimation function
        
        The proximal estimation function as 
        described in the base class :class:`vampyre.estim.base.Estim`
                
        :param r: Proximal mean
        :param rvar: Proximal variance
        :param Boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        
        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior
            mean, variance and optional cost.
        """              
        
        # Save parameters for the optimization
        self.r0_cg, self.r1_cg = r
        rvar0, rvar1 = rvar
        self.rvar0_cg = common.repeat_axes(
                        rvar0, self.shape0, self.z0rep_axes, rep=False)        
        self.rvar1_cg = common.repeat_axes(
                        rvar1, self.shape1, self.z1rep_axes, rep=False)
        self.wvar_cg = common.repeat_axes(
                        self.wvar, self.shape1, self.wrep_axes, rep=False)
                        
        # Get dimensions
        self.n0 = np.prod(self.shape0)
        self.n1 = np.prod(self.shape1)
        
        """
        First-order terms
        """
        # Run the CG optimization to compute the first order terms
        zinit = self.pack(self.r0_cg,self.r1_cg)
        zvec = scipy.optimize.fmin_cg(self.feval, zinit, self.fgrad,disp=0,\
            maxiter=self.nit_cg)
        zhat = self.unpack(zvec)
        
        """
        Cost
        """
        if return_cost:
            # Compute the cost                
            cost = self.feval(zvec)
    
            # Add the cost for the second order terms.
            # 
            # We only consider the MAP case, where the second-order cost is
            # (1/2)*nz1*log(2*pi*wvar)
            if self.is_complex:
                cscale = 1
            else:
                cscale = 2
            if self.map_est:
                if np.all(self.wvar > 0):
                    cost += (1/cscale)*self.n1*np.mean(np.log(cscale*np.pi*self.wvar))                

        """
        Second-order terms
        """        
        # For the second order terms, we first compute the numerical gradient
        # along a random direction
        step = 1e-2;
        dr0 = np.random.normal(0,step,self.shape0)*np.sqrt(self.rvar0_cg)
        dr1 = np.random.normal(0,step,self.shape1)*np.sqrt(self.rvar1_cg)
        self.r0_cg += dr0
        self.r1_cg += dr1        
        zvec1 = scipy.optimize.fmin_cg(self.feval, zvec, self.fgrad,disp=0,\
            maxiter=self.nit_cg)        
        dzvec = zvec1 - zvec
        dz0, dz1 = self.unpack(dzvec)
        
        # Then, the variance is given by gradient
        alpha0 = np.mean(np.real(dr0.conj()*dz0),self.z0rep_axes) / \
                 np.mean(np.abs(dr0)**2,self.z0rep_axes)
        alpha1 = np.mean(np.real(dr1.conj()*dz1),self.z1rep_axes) / \
                 np.mean(np.abs(dr1)**2,self.z1rep_axes)
        zhatvar0 = alpha0*rvar0
        zhatvar1 = alpha1*rvar1
        zhatvar = [zhatvar0, zhatvar1]
                                                         
        if return_cost:
            return zhat,zhatvar, cost
        else:
            return zhat,zhatvar                    

        return 0
        
    def unpack(self,zvec):
        """
        Unpacks the variables from vector for the CG estimation
        """
        z0 = zvec[:self.n0].reshape(self.shape0)
        z1 = zvec[self.n0:].reshape(self.shape1)
        return z0,z1
        
    def pack(self,z0,z1):
        """
        Packs the variables from vector for the CG estimation to a vector
        """        
        zvec = np.hstack((z0.ravel(), z1.ravel()))
        return zvec
            
    def feval(self,z):
        """
        The objective function for the CG optimization
        """
        z0, z1 = self.unpack(z)        
        cost0 = np.sum((np.abs(z0-self.r0_cg)**2)/self.rvar0_cg)
        cost1 = np.sum((np.abs(z1-self.r1_cg)**2)/self.rvar1_cg)
        costm = np.sum((np.abs(z1-self.A.dot(z0))**2)/self.wvar_cg)
        return (cost0+cost1+costm)/2
    
    def fgrad(self,z):
        z0, z1 = self.unpack(z)
        g0 = (z0-self.r0_cg)/self.rvar0_cg
        g1 = (z1-self.r1_cg)/self.rvar1_cg
        e = (z1-self.A.dot(z0))/self.wvar_cg
        g0 = g0 - self.A.dotH(e)
        g1 = g1 + e
        grad = self.pack(g0,g1)
        return grad
        
        
    def est_svd(self,r,rvar,return_cost=False):
        """
        SVD-based estimation function
        
        The proximal estimation function as 
        described in the base class :class:`vampyre.estim.base.Estim`
                
        :param r: Proximal mean
        :param rvar: Proximal variance
        :param Boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        
        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior
            mean, variance and optional cost.
        """        
        
        # Unpack the variables
        r0, r1 = r
        rvar0, rvar1 = rvar
        
        
        # Get the diagonal parameters
        s, sshape, srep_axes = self.A.get_svd_diag()        
        
        # Get dimensions
        nz1 = np.prod(self.shape1)
        nz0 = np.prod(self.shape0)
        ns = np.prod(sshape)
        
        # Reshape the variances to match the dimensions of the first-order
        # terms
        s_rep = common.repeat_axes(s, sshape, srep_axes, rep=False)
        rvar0_rep = common.repeat_axes(
                        rvar0, self.shape0, self.z0rep_axes, rep=False)        
        rvar1_rep = common.repeat_axes(
                        rvar1, self.shape1, self.z1rep_axes, rep=False)
        wvar_rep = common.repeat_axes(
                        self.wvar, sshape, self.wrep_axes, rep=False)

        """
        To compute the estimates, we write:
        
            z0 = V*q0 + z0perp,  z1 = U*q1 + z1perp.
            
        We separately compute the estimates of q=(q0,q1), z0perp and z1perp.
        
        First, we compute the estimates for q=(q0,q1).  The joint density
        of q is given by p(q) \propto exp(-J(q)) where
        
            J(q) = ||q1-s*q0 - bt||^2/wvar + ||q1-q1bar||^2/rvar1 
                 + ||q0-q0bar||^2/rvar0
        where q0bar = V'(r0), q1bar = U'(z1), bt = U'(b).
        
        Now define c := [-s, 1]  P^{-1} = diag(rvar0,rvar1)
        Hence 
            J(q) = ||c'q-bt||^2/wvar + (q-qbar)'P^{-1}(q-qbar)
                 = q'Q^{-1}q -2q'*g
        where
            Q = (P^{-1} - cc'/wvar)^{-1}  = P - Pcc'P/(wvar + c'Pc)
            g = bt*c/wvar + P^{-1}qbar
        
        Now, we can verify the following:
           Q = cov(q)
           det(Q) = d = 1/(wvar + rvar1 + |s|^2*rvar0)
           Q*c/wvar = d*P^{-1}c = d*[rvar0*s, rvar1]
           Q*P^{-1}*qbar = qbar - Pc d*c'*qbar
               = qbar - [-rvar0*s,rvar1]*d*(qbar1-s*qbar0)
        Hence 
           qhat = E(q) = qbar - [-rvar0*s,rvar1]*d*(qbar1-s*qbar0-bt)        
        """
        
        # Infinite variance case
        if np.any(rvar1 == np.Inf):
            zhat0 = r0
            zhatvar0 = rvar0
            zhat1 = self.A.dot(r0) + self.b
            qvar1 = np.mean(np.abs(s_rep)**2*rvar0_rep+wvar_rep,self.z1rep_axes)
            zhatvar1 = ns/nz1*qvar1 + (1-ns/nz1)*self.wvar
            cost = 0  # FIX THIS
            zhat = [zhat0,zhat1]
            zhatvar = [zhatvar0,zhatvar1]
            if return_cost:
                return zhat,zhatvar, cost
            else:
                return zhat,zhatvar                    
        
        
        # Compute the offset terms
        qbar0 = self.A.VsvdH(r0)
        qbar1 = self.A.UsvdH(r1)

        # Compute E(q)    
        d = 1/((np.abs(s_rep)**2)*rvar0_rep + rvar1_rep + wvar_rep)    
        e = d*(qbar1-s_rep*qbar0-self.bt)
        qhat0 = qbar0 + rvar0_rep*s_rep*e
        qhat1 = qbar1 - rvar1_rep*e
        
        """
        Compute E(z)
        
        zhat0 = (I-VV')*r0 + V*qhat0 = r0 + V*(qhat0-qbar0)
        zhat1 = (I-UU')*(wvar*r1+rvvar1*b)/(wvar+rvar0) + U*qhat1
              = 
        """
        
        # Compute E(z)
        z1p = (wvar_rep*r1 + rvar1_rep*self.b)/(wvar_rep+rvar1_rep)
        qbar1p = (wvar_rep*qbar1 + rvar1_rep*self.bt)/(wvar_rep+rvar1_rep)
        zhat0 = r0 + self.A.Vsvd(qhat0-qbar0)
        zhat1 = z1p + self.A.Usvd(qhat1-qbar1p)
        zhat = [zhat0,zhat1]
                
        """
        Compute the variance.
        From the above calcualtions, we have
            cov(q) = Q = P - Pcc'P/(wvar + c'Pc)
        
        var(q0) = rvar0 - d*(rvar0^2)|s|^2 = rvar0*(1-d*rvar0*|s|^2)
        var(q1) = rvar1 - d*(rvar1^2) = rvar1*(1-d*rvar1)
        """
        qvar0 = rvar0_rep*(1-d*rvar0_rep*np.abs(s_rep)**2)
        qvar1 = rvar1_rep*(1-d*rvar1_rep)
        qvar0 = np.mean(qvar0, axis=self.z0rep_axes)
        qvar1 = np.mean(qvar1, axis=self.z1rep_axes)
        
        """
        Compute the variance of z
        """
        zhatvar0 = ns/nz0*qvar0 + (1-ns/nz0)*rvar0
        zhatvar1 = ns/nz1*qvar1 + (1-ns/nz1)*rvar1*self.wvar/(self.wvar+rvar1)
        zhatvar = [zhatvar0, zhatvar1]                        

        if not return_cost:
            return zhat, zhatvar
        
        """
        Compute costs from the first order terms:
        
        cost1_perp = min_{z1} ||(I-UU')*(b-z1)||^2/wvar + ||(I-UU')*(r-z1)||^2/rvar1
              = ||(I-UU')*(b-r1)||^2/(wvar+rvar1)
              = ||b-r1 - U*(bt-qbar1)||^2/(wvar+rvar1)
        costq = ||q1-s*q0-bt||^2/wvar              
        cost0_perp = 0        
        """
        e = (self.b-r1-self.A.Usvd(self.bt-qbar1))
        cost1_perp = np.sum((np.abs(e)**2)/(wvar_rep+rvar1_rep))
        if np.all(self.wvar > 0):
            e = qhat1-s_rep*qhat0-self.bt
            costq = np.sum((np.abs(e)**2)/wvar_rep)
        cost0 = np.sum((np.abs(qhat0-qbar0)**2)/rvar0_rep)
        cost1 = np.sum((np.abs(qhat1-qbar1)**2)/rvar1_rep)
        cost = cost1_perp + costq + cost0 + cost1
        
        """
        Compute the costs for the second-order terms.
        
        For the MAP case, cost = -nz1*log(2*pi*wvar)
        
        For the MMSE case, we compute the Gaussian entropies:
            H1p = H((I-UU')*z1) - (nz-ns)log(2*pi*wvar)
            Hq  = H(q) - ns*log(2*pi*wvar)
            H0q = H((I-VV')*z0) 
            cost = cost - H1p - Hq - H0q            
        """
        if self.is_complex:
            cscale = 1
        else:
            cscale = 2
        if self.map_est:
            if np.all(self.wvar > 0):
                cost += nz1*np.mean(np.log(cscale*np.pi*self.wvar))
        else:
            a = cscale*np.pi
            H1p = (nz1-ns)*np.mean(np.log(rvar1/(rvar1+self.wvar)))
            Hq  = ns*np.mean(np.log(a*rvar1_rep*rvar0_rep*d))
            H0p = (nz0-ns)*np.mean(np.log(a*rvar0))
            cost = cost - H1p - Hq - H0p
        
        # Scale by 2 for the real case
        cost /= cscale
        
        return zhat, zhatvar, cost
        

