# -*- coding: utf-8 -*-
"""
"""

# Add the path to the vampyre module and then load the module
import sys
import os
vp_path = os.path.abspath('../')
if not vp_path in sys.path:
    sys.path.append(vp_path)
import vampyre as vp

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
       ['solver.vamp.vamp_test_mult',vp.solver.vamp.vamp_test_mult],\
       ['estim.linear_two.lin_two_test_mult',vp.estim.linear_two.lin_two_test_mult]\
    ]
    
    cnt = 0
    cnt_pass = 0
    for test in tests:
        name = test[0]
        fn = test[1]
        cnt += 1
        try:
            fn()
            print(name+" Pass")
            cnt_pass += 1
        except vp.common.utils.TestException as err:
            print(name+" Fail")      
            print(err.msg)
    
    print("{0:d} out of {1:d} passed".format(cnt_pass,cnt))
            
                
if __name__ == "__main__":
    test_version()
