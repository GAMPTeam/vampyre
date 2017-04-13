"""
env.py:  Sets environment variables including path
"""
import os
import sys

# ensure some directories are in our PYTHONPATH, ( relative to test dir)
def add_vp_path():
    for d in ('..','..\..'):
        if sys.version[0] == '2':
            fd = d
        else:
            fd = os.path.abspath( os.path.dirname(__file__) + os.path.sep + d )
            #fd = os.path.abspath( d )
        if not fd in sys.path:
            sys.path.append(fd)
