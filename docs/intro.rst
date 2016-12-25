.. _page_intro:

Introduction
============

The :mod:`vampyre` package provides methods to formulate and solve
classes of problems described by linear constraints.  As a simple example,
consider the problem of estimating a unknown vector :math:`z` via an 
optimization of the form,

.. math::

     \min_z F(z), \quad  F(z) = \phi(z) + \frac{1}{2}\|y-Az\|^2,

where :math:`y` is a known vector, :math:`A` is a known linear
transform and :math:`\phi(z)` is some penalty function.  In statistics,
this problem arises in linear regression where 
:math:`z` are regression coefficients,  :math:`A` is a transformed
data matrix and :math:`\phi(z)` is a regularizer.
In signal processing, the optimizations is used for linear inverse problems,
where :math:`z` may represent an image or some other object to be
reconstructed,  :math:`A` is an observation
transform (liking a blurring operation), and :math:`\phi(z)` is used
to constrain the object in the reconstruction.  In compressed sensing,
the function :math:`\phi(z)=\lambda\|z\|_1` to impose sparsity on 
:math:`z`.

This simple regularized least squares optimization has the basic ingridients
of the problems solved by the :mod:`vampyre` package:  unknown vectors, 
factorizable costs functions and linear constraints.    