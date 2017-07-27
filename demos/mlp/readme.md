
# ML-VAMP demo for inference in multi-layer perceptron

The programs in this directory reproduce the results in the paper, 

* A.K.Fletcher, S.Rangan, "Inference in Deep Networks in High Dimensions", 
[arxiv preprint](https://arxiv.org/abs/1706.06549)

There are two sets of programs:
* Inference on a synthetic random MLP, where the goal is to estimate the
  input of the MLP from the output.
* Use of the above inference method for inpainting reconstruction of MNIST
  digits.
  
## Inference in a Synthetic Random MLP

In this test, a random MLP is generated and the ML-VAMP algorithm is
used to estimate the hidden states in that network.
You will need to perform the following steps:

* Run `python mlp_test.py`.  This will run an outer loop that varies the
number of measurements (`ny`), and for each value of `ny`, the program
will run a number of trials.  In each trial, it will generate a 
random network, run the ML-VAMP algorithm and record the actual and
predicted MSE of the estimate.  The results are then stored in a `pickle` file,
`randmlp_sim.pkl`.  The number of trials per measurements and the number
of measurements can be set with the `-nit` and `-ny` options.

* The results of the simulation can then be plotted via the command, 
`python mlp_plot.py [-plt_iter] [-plt_meas]`.  With the `-plt_iter`
option, the program will create a PNG figure for the MSE as a function
of the iteration.  with the `-plt_meas`, it will create a PNG figure 
of the final MSE as a function of the number of measurements (`ny`).




