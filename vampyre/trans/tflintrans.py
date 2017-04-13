"""
tflintrans.py:  Linear transforms based on Tensorflow ops
"""
from __future__ import division

import tensorflow as tf
import numpy as np

# Import other subpackages in vampyre
import vampyre.common as common

# Import individual classes from same modules in the same package
from vampyre.trans.base import LinTrans

class TFLinTrans(LinTrans):
    """
    Linear transform class based on Tensorflow operations
    
    :param x_op:  TF op for the input
    :param y_op:  TF op for the output
    :param sess:  TF session to run the operator in
    :param Boolean remove_bias:  For affine operators, this removes the bias
    
    :note:  It is assumed that the computation graph from `x_op` to `y_op` is
       linear (possibly affine).
    """    
    def __init__(self,x_op,y_op,sess,remove_bias=False):
        # Save parameters
        self.x_op = x_op
        self.y_op = y_op
        self.sess = sess
        self.remove_bias = remove_bias

        # Get dimensions and data types
        self.shape0 = x_op.get_shape()
        self.shape1 = y_op.get_shape()
        self.dtype0 = x_op.dtype
        self.dtype1 = y_op.dtype
                
        # Create the ops for the gradient.  If the linear operator is y=F(x),
        # then z = y'*F(x).  Therefore, dz/dx = F'(y).
        self.ytr_op = tf.placeholder(self.dtype1,self.shape1)        
        self.z_op = tf.reduce_sum(tf.multiply(tf.conj(self.ytr_op),self.y_op))
        self.zgrad_op = tf.gradients(self.z_op,self.x_op)[0]
        
        # Compute output at zero to subtract 
        if self.remove_bias:
            xzero = np.zeros(self.shape0)
            self.y_bias = self.sess.run(self.y_op, feed_dict={self.x_op: xzero})
        else:
            self.y_bias = 0
                
    
    def dot(self,x):
        """
        Forward multiplication
        """
        y = self.sess.run(self.y_op, feed_dict={self.x_op: x})
        if self.remove_bias:
            y -= self.y_bias
        return y
        
    def dotH(self,ytr):
        """
        Adjoint multiplication.  This is computed as a gradient of z_op
        """
        xtr = self.sess.run(self.zgrad_op, \
            feed_dict={self.ytr_op: ytr})
        xtr = np.reshape(xtr, self.shape0)
        return xtr
    