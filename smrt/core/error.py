"""
Definition of the Exception specific to SMRT.
"""

import warnings


class SMRTError(Exception):
    """
    Error raised by the model.
    """

    pass


class SMRTWarning(Warning):
    """
    Warning raised by the model.
    """

    pass


def smrt_warn(message, **kwargs):
    """
    Issue a SMRT warning with a standard message about disabling warnings.
    """

    disable_warning_message = """To disable all smrt warnings, use: import warnings;
warnings.filterwarnings("ignore", smrt.error.SMRTWarning). See the warnings Python documentation for finer
controls.

"""
    if "stacklevel" in kwargs:
        kwargs["stacklevel"] += 1
    else:
        kwargs["stacklevel"] = 2
    warnings.warn(message + "\n\n" + disable_warning_message, category=SMRTWarning, **kwargs)
