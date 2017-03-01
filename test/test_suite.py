"""
test_suite.py:  Runs through tests
"""
from __future__ import print_function

# Add the path to the vampyre module and then load the module
import sys

# Ensure some directories are in our PYTHONPATH, 
# (Should be relative to project root w/ PyTest -- e.w.t. 3/1/17)
# The directory search does not appear to work in python 2.
for d in ('./test','./test/sparse'):    # 
    # Removed the verison check, don't need the full path -- e.w.t.
    if not d in sys.path:
        sys.path.append(d)
        
import vampyre as vp
import vamp_sparse_test

def test_version():
    vp.version_info()
    
    # Unit tests
    tests = [\
       ['common.utils.repeat_test', vp.common.utils.repeat_test], \
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
            
                
if __name__ == "__main__":
    test_version()
