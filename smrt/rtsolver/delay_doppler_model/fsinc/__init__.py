import numpy as np

# commented by Ghislain
# from .fsinc_1d_u import *
# from .fsinc_1d_nu import *

# from .fsinc_2d_u import *
# from .fsinc_2d_nu import *


def zero_offset(*x):
    """Moves arrays provided in x to be offset to zero, using the first array as reference."""
    minx = np.min(x[0])
    return tuple([(xx - minx) for xx in x])
