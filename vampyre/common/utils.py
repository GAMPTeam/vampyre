"""
utils.py
"""
import numpy as np

class VpException(Exception):
    """
    Basic error in the Vampyre package
    
    :param string str:  Error message.
    """
    def __init__(self, msg):
        self.msg = msg
        

class TestException(VpException):
    """
    Used to indicate failure in a unit test
    """
    def __init__(self,msg):
        VpException.__init__(self,msg)
        

def repeat_axes(u,shape,rep_axes,rep=True):
    """
    Repeats an array over multiple axes
    
    Given an input array :code:`u` the function
    returns an array of size :param:`shape where the
    data is specified on the axes :param:`rep_axes` 
    
    :param u:  Input array
    :param shape:  Shape of output array
    :param rep_axes:  Axes on which :code:`u` is to be 
               repeated.
    :param Boolean rep:  If set to :code:`False`, the
               array is only resized to having a dimension
               1 on the repeated axes.  This is useful if 
               the result will be used with python 
               broadcasting.  (default=:code:`True`)
    :returns: :code:`urep`, the repeated array                 
    """
    
    # Find the axes to repeat
    ndim = len(shape)
    shape0 = np.ones(ndim,dtype=int)
    axes_spec = [i for i in range(ndim) if i not in rep_axes]
    
    # Reshape the array u to the size with 1's in the
    # dimensions to be repeated
    shape0[axes_spec] = np.array(shape,dtype=int)[axes_spec]
    urep = np.reshape(u,shape0)
    
    # Repeat the matrix, if required
    if rep:
        for i in rep_axes:
            urep = np.repeat(urep,shape[i],axis=i)
    return urep                
    
def repeat_sum(u,shape,rep_axes):
    """
    Computes sum of a repeated matrix
    
    In effect, this routine computes 
    code:`np.sum(repeat(u,shape,rep_axes))`.  However, it performs
    this without having to perform the full repetition.
    
    """
    # Must convert to np.array to perform slicing
    shape_vec = np.array(shape,dtype=int)
    rep_vec = np.array(rep_axes,dtype=int)
    
    # repeat and sum
    urep = repeat_axes(u,shape,rep_axes,rep=False)
    usum = np.sum(urep)*np.product(shape_vec[rep_vec])
    return usum
        
    
# def repeat_test(verbose=False, tol=1e-8, raise_exception=True):
#     """
#     Unit test for the repeat_mult function.

#     :param Boolean verbose:  Print test results    
#     :param tol:  Tolerance for passing test
#     :param Boolean raise_exception:  Raises an error on test failure.  This 
#         can be caught in the unit test dispatcher.
#     """    
#     ushape = (2,2,3)
#     rep_axes = (1,2)
#     u = np.random.uniform(0,1,size=ushape)
#     uavg = np.mean(u, axis=rep_axes)
#     urep = repeat_axes(uavg,shape=ushape,rep_axes=rep_axes)
    
#     # Makse sure that the sums are the same
#     d = 0
#     for i in range(ushape[0]):
#         sum0 = np.sum(u[i,:,:])
#         sum1 = np.sum(urep[i,:,:])
#         d += np.abs(sum1-sum0)
#     if verbose:
#         print("total diff= {0:f}".format(d))
                
#     if (d > tol) and raise_exception:
#         raise TestException("Sum along repeated matrix is not conistent")
    
    
                
