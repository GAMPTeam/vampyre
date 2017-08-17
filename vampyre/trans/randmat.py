"""
randmat.py:  Random matrix methods
"""
from __future__ import division

import numpy as np

def rand_rot_invariant_mat(nz1,nz0,cond_num=10,is_complex=False):
    """
    Creates a rotationally invariant random matrix.
    
    A rotationally invariant matrix is of the form :math:`A=USV^*` where
    :math:`U` and :math:`V` are uniformly distributed on the unitaries 
    (for complex matrices) and orthogonal matrices (for real matrices).
    The singular values :math:`S=diag(s_1,\ldots,s_r)` are logarithmically
    spaced from :math:`1/cond_num` to 1.  The singular values are then 
    scaled to have an average magnitude squared of one.
    """
    
    # Generate a random Gaussian matrix
    if is_complex:
        A = np.random.randn(nz1,nz0) + 1j*np.random.randn(nz1,nz0)
    else:
        A = np.random.randn(nz1,nz0)
        
    # Take the SVD
    U,s,V = np.linalg.svd(A,full_matrices=0)

    # Reset the singular values    
    r = len(s)
    s = np.logspace(-np.log10(cond_num),0,r)
    s = s / np.sqrt(np.mean(s**2))
    
    # Rebuild the matrix
    A = (U*s[None,:]).dot(V)
        
    return A
    

    