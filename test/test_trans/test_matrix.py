"""
test_matrix.py:  Test suite for the linear transform :class:`MatrixLT`
"""
from __future__ import print_function, division

import unittest
import numpy as np

# Add the path to the vampyre package and import it
import env
env.add_vp_path()
import vampyre as vp

def matrix_test(Ashape=(50,100),shape0=(100,10), tol=1e-8, verbose=False,\
                raise_exception=True):
    """
    Unit test for the :class:`MatrixLT` class.
    
    :param Ashape:  Shape of matrix
    :param shape0:  Shape of input
    :param Boolean verbose:  Print results of test.
    :param tol:  Tolerance for passing test
    :param Boolean raise_exception:  Raises an error on test failure.  This 
        can be caught in the unit test dispatcher.
    """
    # Generate random matrix, input and output
    A = np.random.uniform(0,1,Ashape)
    z0 = np.random.uniform(0,1,shape0)
    z1 = A.dot(z0)

    # Create corresponding matrix operator
    Aop = vp.trans.MatrixLT(A,shape0)

    # Perform the multiplication via SVD
    s = Aop.get_svd_diag()[0]     
    q0 = Aop.VsvdH(z0)
    q1 = Aop.svd_dot(s,q0)
    z1est = Aop.Usvd(q1)
    
    # Test fails if SVD method does not match direct method
    err = np.linalg.norm(z1-z1est)
    if verbose:
        print("Error: {0:12.4e}".format(err))        
    if (err > tol) and raise_exception:
        raise vp.common.TestException("SVD method for performing multiplication"\
            +"does not match direct multiplication.  "\
            +"err={0:12.4e}".format(err))
        
           

class TestCases(unittest.TestCase):
    def test_matrix(self):
        """
        Tests the matrix.
        """        
        matrix_test()
            
        
if __name__ == '__main__':    
    unittest.main()
    
