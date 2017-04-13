# Load sub-packages
import vampyre.common
import vampyre.estim
import vampyre.trans
import vampyre.solver

def version():
    """
        Return the current version string for the vampyre package.
        
        >>> version()
        '0.0'
    """
    return "0.0"

def version_info():
    print("vampyre version " + version())