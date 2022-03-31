"""Definition of the Exception specific to SMRT.

"""

import warnings


class SMRTError(Exception):
    """Error raised by the model"""
    pass


class SMRTWarning(Warning):
    """Warning raised by the model"""
    pass


def smrt_warn(message):
    warnings.warn(message, category=SMRTWarning, stacklevel=2)
