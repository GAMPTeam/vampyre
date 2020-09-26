"""
linear.py:  Linear estimation class
"""
from __future__ import division

import numpy as np

# Import other subpackages in vampyre
import vampyre.common as common
import vampyre.trans as trans

# Import individual classes and methods from the current sub-package
from vampyre.estim.base import BaseEst


class LinEst(BaseEst):
    """
    Estimator based on a linear constraint with noise

    This estimator corresponds to a linear constraint of the form
    :math:`y = Az + w`
    where :math:`w \\sim {\\mathcal N}(0,\\tau_w I)`.
    Specifically, the penalty is

    :math:`f(z) = (1/2\\tau_w)\|y-Az\|^2 + (d/2)\\ln(2\\pi \\tau_w)`,

    where :math:`d` is the dimension of :math:`y`

    :param A: Linear operator represented as a
        :class:`vampyre.trans.base.LinTrans`
    :param y:  Output
    :param wvar:  Noise level
    :param wrep_axes':  The axes on which the output noise variance is repeated.
        Default is 'all'.
    :param var_axes:  The axes on which the input variance is averaged.
    :param tune_wvar:  Enables tuning of noise level.  In this case,
        :code:`wvar` is used as an initial condition.
    :param Boolean is_complex:  indiates if :math:`z` is complex
    :param Boolean map_est:  indicates if estimator is to perform MAP
        or MMSE estimation. This is used for the cost computation.
    :param rvar_init:  Initial prior variance used in the
        :code:`est_init` method.

    :note:  The linear operator :code:`A` must have :code:`svd_avail==True`.
       In the future, if an SVD is not available, we will use conjugate
       gradient
    :note:  The linear operator must also have the :code:`shape0` and
       :code:`shape1` arrays available to compute the dimensions.

    :note:  The axes :code:`wrep_axes` and :code:`zerp_axes` must
       include the axis in which :code:`A` operates.
    """
    def __init__(self,A,y,wvar=0,\
                 wrep_axes='all', var_axes=(0,),name=None,map_est=False,\
                 is_complex=False,rvar_init=1e5,tune_wvar=False):

        BaseEst.__init__(self, shape=A.shape0, var_axes=var_axes,\
            dtype=A.dtype0, name=name,\
            type_name='LinEstim', nvars=1, cost_avail=True)
        self.A = A
        self.y = y
        self.wvar = wvar
        self.map_est = map_est
        self.is_complex = is_complex
        self.cost_avail = True
        self.rvar_init = rvar_init
        self.tune_wvar = tune_wvar

        # Get the input and output shape
        self.zshape = A.shape0
        self.yshape = A.shape1

        # Set the repetition axes
        ndim = len(self.yshape)
        if wrep_axes == 'all':
            wrep_axes = tuple(range(ndim))
        self.wrep_axes = wrep_axes

        # Compute the SVD terms
        # Take an SVD A=USV'.  Then write p = SV'z + w,
        if not A.svd_avail:
            raise common.VpException("Transform must support an SVD")
        self.p = A.UsvdH(y)
        srep_axes = A.get_svd_diag()[2]

        # Compute the norm of ||y-UU*(y)||^2/wvar
        if np.all(self.wvar > 0):
            yp = A.Usvd(self.p)
            wvar1 = common.repeat_axes(wvar, self.yshape, self.wrep_axes, rep=False)
            err = np.abs(y-yp)**2
            self.ypnorm = np.sum(err/wvar1)
        else:
            self.ypnorm = 0

        # Check that all axes on which A operates are repeated
        for i in range(ndim):
            if not (i in self.wrep_axes) and not (i in srep_axes):
                raise common.VpException(
                    "Variance must be constant over output axis")
            if not (i in self.var_axes) and not (i in srep_axes):
                raise common.VpException(
                    "Variance must be constant over input axis")


    def est_init(self, return_cost=False, ind_out=None,\
        avg_var_cost=True):
        """
        Initial estimator.

        See the base class :class:`vampyre.estim.base.Estim` for
        a complete description.

        :param boolean return_cost:  Flag indicating if :code:`cost` is
            to be returned
        :returns: :code:`zmean, zvar, [cost]` which are the
            prior mean and variance
        """
        # Check parameters
        if (ind_out != [0]) and (ind_out != None):
            raise ValueError("ind_out must be either [0] or None")
        if not avg_var_cost:
            raise ValueError("disabling variance averaging not supported for LinEst")

        # Get the diagonal parameters
        s, sshape, srep_axes = self.A.get_svd_diag()
        shape0 = self.A.shape0

        # Reshape the variances to the transformed space
        s1    = common.repeat_axes(s, sshape, srep_axes)
        wvar1 = common.repeat_axes(self.wvar, sshape, self.wrep_axes, rep=False)

        # Compute the estimate within the transformed space
        q = (1/s1)*self.p
        qvar = wvar1/(np.abs(s1)**2)
        qvar_mean = np.mean(qvar, axis=self.var_axes)

        rdim = np.product(sshape)/np.product(shape0)
        zmean = self.A.Vsvd(q)
        zvar = rdim*qvar_mean + (1-rdim)*self.rvar_init

        # Exit if cost does not need to be computed
        if not return_cost:
            return zmean, zvar

        # Computes the MAP output cost
        if np.all(self.wvar > 0):
            cost = self.ypnorm
        else:
            cost = 0

        # Compute the output variance cost
        if np.all(self.wvar > 0) and self.map_est:
            clog = np.log(2*np.pi*self.wvar)
            cost += common.repeat_sum(clog, self.zshape, self.wrep_axes)

        # Scale for real case
        if not self.is_complex:
            cost = 0.5*cost
        return zmean, zvar, cost


    def est(self,r,rvar,return_cost=False, ind_out=None,\
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
            raise ValueError("disabling variance averaging not supported for LinEst")


        # Get the diagonal parameters
        s, sshape, srep_axes = self.A.get_svd_diag()

        # Get dimensions
        nz = np.prod(self.zshape)
        ny = np.prod(self.yshape)
        ns = np.prod(sshape)

        # Reshape the variances to the transformed space
        s1    = common.repeat_axes(s, sshape, srep_axes, rep=False)
        rvar1 = common.repeat_axes(rvar, sshape, self.var_axes, rep=False)
        wvar1 = common.repeat_axes(self.wvar, sshape, self.wrep_axes, rep=False)

        # Compute the estimate within the transformed space
        qbar = self.A.VsvdH(r)
        d = 1/(rvar1*(np.abs(s1)**2) + wvar1)
        q = d*(rvar1*s1.conj()*self.p + wvar1*qbar)
        qvar = rvar1*wvar1*d
        qvar_mean = np.mean(qvar, axis=self.var_axes)

        zhat = self.A.Vsvd(q - qbar) + r
        zhatvar = ns/nz*qvar_mean + (1-ns/nz)*rvar

        # Update the variance estimate if tuning is enabled
        if self.tune_wvar:
            yerr = np.abs(self.y - self.A.Usvd(s1*q))**2
            self.wvar = np.mean(yerr, self.wrep_axes) + np.mean(qvar*(np.abs(s1)**2),self.wrep_axes)

        # Exit if cost does not need to be computed
        if not return_cost:
            return zhat, zhatvar

        # Computes the MAP output cost
        if np.all(self.wvar > 0):
            err = np.abs(self.p-s1*q)**2
            cost = self.ypnorm + np.sum(err/wvar1)
        else:
            cost = 0

        # Add the MAP input cost
        err = np.abs(q-qbar)**2
        cost = cost + np.sum(err/rvar1)

        # Compute the variance cost.
        if self.map_est:
            # For the MAP case, this is log(2*pi*wvar)
            if np.all(self.wvar > 0):
                cost += ny*np.mean(np.log(2*np.pi*self.wvar))
        else:
            # For the MMSE case, this is 1 + log(2*pi*wvar) - H(b)
            # where b is the Gaussian with variance wvar*rvar*d
            cost +=  -ns*np.mean(np.log(rvar1*d)) -\
                (nz-ns)*np.mean(np.log(2*np.pi*rvar1))
            if np.all(self.wvar > 0):
                cost += (ny-ns)*np.mean(np.log(2*np.pi*self.wvar))

        # Scale for real case
        if not self.is_complex:
            cost = 0.5*cost

        return zhat, zhatvar, cost
