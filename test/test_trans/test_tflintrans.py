"""
test_tflintrans.py:  Test suite for the linear estimator :class:`TFLinTrans`
"""
from __future__ import print_function, division

# Removes the warning that "Tensorflow library was not compiled to use SSE..."
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import unittest
import numpy as np

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp


def conv2d_test(xshape=(5,32,32,3), nchan_out=4, verbose=False, diff_tol=1e-3):
    """
    Unit test for the TFLinTrans class based on using the `conv2d` TF op
    
    :param xshape:   Input shape.  This should be a 4-dim tuple
    :param nchan_out:  Number of channels in the output
    :param Booolean verbose:  Boolean flag indicating if the
       results are to be displayed
    :param diff_tol:  Tolerance level above which test fails
    """    

    # Get dimensions
    nchan_in = xshape[3]
    wx = 5        # Filter dimension in x dimension
    wy = 5        # Filter dimension in y dimension
    
    # Create op for the input
    tf.reset_default_graph()    
    x_op = tf.placeholder(tf.float32, xshape,name='x')

    # Create random filter
    wshape = (wx,wy,nchan_in,nchan_out)
    W = np.random.normal(0,1/np.sqrt(wx*wy*nchan_in),wshape)    
    W_op = tf.Variable(W,dtype=tf.float32,name='W')
    
    # Create output through filtering 
    y_op = tf.nn.conv2d(x_op,W_op,strides=[1,2,2,1],padding='SAME', name='y')
    
    # Open the session
    sess = tf.Session()
        
    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    
    # Create the linear operator
    lin_op = vp.trans.TFLinTrans(x_op,y_op,sess)
    
    # Create random inputs and outputs
    z0 = np.random.normal(0,1,lin_op.shape0)
    u1 = np.random.normal(0,1,lin_op.shape1)
    
    # Test the operator by verifying that the inner products are identical:
    #   <u1,F.dot(z0) >  =  <F.dotH(u1), z0 >
    z1 = lin_op.dot(z0)
    u0 = lin_op.dotH(u1)
    
    # Check failure condition
    ip1 = np.sum(z1*u1)
    ip2 = np.sum(z0*u0)
    diff = np.abs(ip2-ip1)
    fail = (diff > diff_tol)
    if verbose or fail:
        print("Inner products: {0:12.4e} {1:12.4e} Diff: {2:12.4e}".format(\
            ip1, ip2, diff))
    if fail:
        raise vp.common.VpException("Inner products do not match")
            

class TestCases(unittest.TestCase):
    def test_tflintrans_conv2d(self):
        """
        Run the conv2d test.
        
        Note that on Windows 10 with tensorflow 1.0, there is a long warning 
        that can be ignored.  This will be fixed in the next TF release.
        """        
        conv2d_test()
            
        
if __name__ == '__main__':    
    unittest.main()
    
