"""
Contains various utilities that work with/for SMRT.

The wrappers to legacy snow radiative transfer models can be used to run DMRT-QMS (passive mode), HUT and MEMLS (passive mode).
Other tools are listed below.

"""

import numpy as np

LOG10 = np.log(10.)


def dB(x):
    """
    Computes the ratio x in dB. Any small value is converted to -200dB.

    Args:
        x: Input value or array.

    Returns:
        The value(s) in dB.
    """
    return 10 * np.log(np.maximum(x, 1e-20)) / LOG10


def invdB(x):
    """
    Computes the dB value x in natural value.

    Args:
        x: Value(s) in dB.

    Returns:
        The value(s) in natural units.
    """
    return 10 ** (x / 10.0)
