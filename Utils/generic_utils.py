import sys
import os
import errno 
import numpy

def mkdir_p(path):
    """
        Creates a directory at given path
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise