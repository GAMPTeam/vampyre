# -*- coding: utf-8 -*-
"""
"""

# Add the path to the vampyre module and then load the module
import sys
import os

# ensure some directories are in our PYTHONPATH, ( relative to test dir)
for d in ('.','..','sparse'):
    fd = os.path.abspath( os.path.dirname(__file__) + os.path.sep + d )
    if not fd in sys.path:
        sys.path.append(fd)

import vampyre as vp
import vamp_sparse_test

def test_version():
    vp.version_info()
    
    # Unit tests
    tests = [\
       ['estim.gaussian.gauss_test', vp.estim.gaussian.gauss_test],\
       ['estim.mixture.mix_test', vp.estim.mixture.mix_test],\
       ['trans.matrix.matrix_test', vp.trans.matrix.matrix_test],\
       ['estim.linear.lin_test_mult',vp.estim.linear.lin_test_mult],\
       ['estim.discrete.discrete_test',vp.estim.discrete.discrete_test],\
       ['estim.linear_two.lin_two_test_mult',vp.estim.linear_two.lin_two_test_mult],\
       ['estim.interval.gauss_integral_test',vp.estim.interval.gauss_integral_test],\
       ['estim.interval.hard_thresh_test',vp.estim.interval.hard_thresh_test],\
       ['estim.relu.relu_test',vp.estim.relu.relu_test],\
       ['solver.vamp.vamp_test_mult',vp.solver.vamp.vamp_test_mult],\
       ['solver.mlvamp.mlvamp_probit_test',vp.solver.mlvamp.mlvamp_probit_test],\
       ['vamp_sparse_test.sparse_inv',vamp_sparse_test.sparse_inv]\
    ]
    
    cnt = 0
    cnt_pass = 0
    for test in tests:
        name = test[0]
        fn = test[1]
        cnt += 1
        sys.stdout.write("{0:40s} ".format(name))
        try:
            fn()
            print("Pass")
            cnt_pass += 1
        except vp.common.utils.TestException as err:
            print("Fail")
            print("   "+err.msg)
    
    print("{0:d} out of {1:d} passed".format(cnt_pass,cnt))
            
                
# if __name__ == "__main__":
#     test_version()
