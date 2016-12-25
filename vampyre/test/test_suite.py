import vampyre as vp

def test_version():
    vp.version_info()
    
    tests = [['common.utils.repeat', vp.common.utils.repeat_test]]
    
    for test in tests:
        name = test[0]
        fn = test[1]
        fn(verbose=True)
        
        print(name+" Passed")
        
    