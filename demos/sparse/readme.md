# AMP for Sparse Linear Inverse Problems

AMP methods had their beginning in solving sparse linear inverse problems
in the area of compressed sensing.  This directory provides examples
to illustrate the AMP methods in simple sparse linear problems on synthetic data.
The examples are a good way to demonstrate how to use the tools in the 
`vampyre` package before undertaking more complex examples.

Four examples are provided:
* [Sparse linear inverse with AMP](sparse_lin_inverse_amp.ipynb) which shows how to use AMP on 
a basic sparse linear inverse problem.  
* [Sparse linear inverse with VAMP](sparse_lin_inverse.ipynb) which shows how to use VAMP on 
the same sparse linear inverse problem.  You can directly start with this demo.
* [Sparse probit](sparse_probit.ipynb) which extends the sparse linear inverse demo
to a nonlinear probit output using ML-VAMP.
* [Sparse linear inverse with EM](sparse_em.ipynb) which adds EM learning to the case
where the statistics on the problem must be learned.


