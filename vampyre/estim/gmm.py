"""
gmm.py:  Estimation methods for a Gaussian mixture model
"""
from __future__ import division
from __future__ import print_function

import numpy as np
from vampyre.estim.base import BaseEst
from vampyre.estim.gaussian import GaussEst
from vampyre.estim.mixture import MixEst
from vampyre.common.utils import VpException, repeat_const

from sklearn import mixture

class GMMEst(BaseEst):
    """ Gaussian mixture estimator class with auto-tuning.
    
    There are two initialization methods.
    If :code:`probc` is supplied (i.e. is not None), 
    then the GMM parameters are initialized by :code:`probc,meanc,varc`.
    Otherwise, they are initialized auto-matically.  With auto-init,
    the user must supply :code:`nc`, :code:`zmean_init` and :code:`zvar_init`.

    :param shape:  shape of the data        
    :param probc:  probability for each cluster  (used as the initial value
        when EM tuning is enabled.)
    :param meanc:  mean for each cluster
    :param varc: variance for each cluster    
    :param is_complex: flag indicating if data is complex
    :param map_est:  flag indicating if the estimator uses MAP estimation.
    :param zvarmin:  minimum variance in each cluster
    :param tune_gmm:  Enables EM tuning of the GMM parameters.
       In this case, `probc`, `meanc` and `varc` are used as initial estimates
        
    :note:  Currently the class only supports scalar (real or complex)
    Gaussian mixtures
    """    
    def __init__(self, shape, probc, meanc, varc,name=None,\
        var_axes=(0,),mean_fix=None,var_fix=None,\
        is_complex=False,map_est=False,zvarmin=1e-3, tune_gmm=False):
            
        # Save properties            
        self.map_est = map_est
        self.is_complex = is_complex
        self.zvarmin = zvarmin
        self.tune_gmm = tune_gmm   
        self.mean_fix = mean_fix
        self.var_fix = var_fix
     
        dtype = meanc.dtype
        BaseEst.__init__(self,shape=shape,var_axes=var_axes,dtype=dtype,\
            name=name, type_name='GMM', nvars=1, cost_avail=True)

        
        # If the GMM parameters are given, set them now
        self.mix = None
        self.set_gmm_param(probc,meanc,varc)         
        self.it = 0
        
        # Set the indicators on which the variance and means are fixed
        if self.mean_fix is None:
            self.mean_fix = np.zeros(len(probc))
        if self.var_fix is None:
            self.var_fix = np.zeros(len(probc))

        
    def set_gmm_param(self,probc,meanc,varc):
        """
        Sets the GMM parameters for the mixture estimator
        """
        nc = len(probc)
                
        if self.mix is None:
            # If the mixture estimator does not exist, create it        
            # First, create the component Gaussian estimators
            est_list = []
            for i in range(nc):
                esti = GaussEst(meanc[i], varc[i], self.shape, 
                     var_axes = self.var_axes, zmean_axes='all',
                     is_complex=self.is_complex, map_est=self.map_est)
                est_list.append(esti)
                
            # Create the mixture 
            self.mix = MixEst(est_list,w=probc)                
            
        else:
            # If the mixture distribution is already created,
            # set the parameters of the mixture estimator
            self.probc = probc
            self.mix.w = np.copy(probc)
            for i in range(nc):
                esti = self.mix.est_list[i]
                if not self.mean_fix[i]:
                    esti.zmean = meanc[i]
                if not self.var_fix[i]:
                    esti.zvar = np.copy(varc[i])
                    
                                 
    def est_init(self, return_cost=False,ind_out=None,\
        avg_var_cost=True):
        """
        Initial estimator.
        
        See the base class :class:`vampyre.estim.base.Estim` for 
        a complete description.
        
        :param Boolean return_cost:  Flag indicating if :code:`cost` is 
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """           
       
        # otherwise, use the mixture estimator
        return self.mix.est_init(return_cost,ind_out,avg_var_cost)
                    
                    
    def est(self,r,rvar,return_cost=False,ind_out=None,\
        avg_var_cost=True):
        """
        Estimation function
        
        The proximal estimation function as 
        described in the base class :class:`vampyre.estim.base.Estim`
                
        :param r: Proximal mean
        :param rvar: Proximal variance
        :param boolean return_cost:  Flag indicating if :code:`cost` is to be returned
        
        :returns: :code:`zhat, zhatvar, [cost]` which are the posterior 
        mean, variance and optional cost.
        """
             
        # Run the estimator with the current parameter settings              
        if return_cost:
            zhat,zhatvar,cost = self.mix.est(r,rvar,return_cost,ind_out,avg_var_cost)
        else:
            zhat,zhatvar = self.mix.est(r,rvar,return_cost,ind_out,avg_var_cost)
            
        # Update the parameters if tuning is enabled
        if (self.tune_gmm):
            self.update_gmm_em()
                    
        
        # Increment the iteation counter
        self.it += 1
            
        # Return the values
        if return_cost:
            return zhat,zhatvar,cost
        else:
            return zhat,zhatvar

    def update_gmm_em(self):
        """
        Updates the GMM parameters using EM estimation
        """
        
        # Get the posterior probabilities, means and variances from the mixture
        # estimator.  The lists have one element for each component in the mixture
        prob_list = self.mix.prob
        zmean_list = self.mix.zmean_list
        zvar_list = self.mix.zvar_list            
        
        # Compute new cluster probabilities mean and variances
        nc = len(prob_list)
        probc = np.zeros(nc)
        meanc = np.zeros(nc)
        varc = []
        for i in range(nc):
            probc[i] = np.mean(prob_list[i])
            meanc[i] = np.mean(prob_list[i]*zmean_list[i])/probc[i]
            dsq = zvar_list[i] + np.abs((zmean_list[i]-meanc[i]))**2
            varci = np.mean((prob_list[i]*dsq)/probc[i],axis=self.var_axes)
            varci = np.maximum(varci, self.zvarmin)
            varc.append(varci)
                

        # Set the parameters
        self.set_gmm_param(probc,meanc,varc)




            