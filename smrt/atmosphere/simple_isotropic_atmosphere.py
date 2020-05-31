# coding: utf-8
"""Implement an isotropic atmosphere with prescribed frequency-dependent emission (up and down) and transmittivity.

 TB and transmissivity can be specified as a constant, or a frequency-dependent dictionary

 To make an atmosphere, it is recommended to use the helper function :py:func:`~smrt.atmosphere.simple_isotropic_atmosphere.make_atmosphere`.


Examples::

    # the full path import is required
    from smrt.atmosphere.simple_isotropic_atmosphere import make_atmosphere

    # Constant
    atmos = make_atmosphere(tbdown=20., tbup=6., trans=1)

    # Frequency-dependent
    atmos = make_atmosphere(tbdown={10e9: 15.2, 21e9: 23.5})

"""

# Stdlib import


# other import
import numpy as np

# local import
from ..core.atmosphere import AtmosphereBase

#
# For the developers:
# A valid atmosphere is any object that provide a tbdown, tbup and trans functions function of frequency and costheta.
# costheta can be an array or a value, both must be handled.
# The implementation below is independent of the propagation angle
#

def make_atmosphere(tbdown=0, tbup=0, trans=1):

    """ Construct an atmosphere instance.

    """

    # create the instance
    return SimpleIsotropicAtmosphere(tbdown=tbdown, tbup=tbup, trans=trans)


class SimpleIsotropicAtmosphere(AtmosphereBase):

    def __init__(self, tbdown=0, tbup=0, trans=1):

        self.constant_tbdown = tbdown
        self.constant_tbup = tbup
        self.constant_trans = trans

    def tbdown(self, frequency, costheta, npol):
        if isinstance(self.constant_tbdown, dict):
            return np.full(len(costheta) * npol, self.constant_tbdown[frequency], dtype=np.float32)
        else:
            return np.full(len(costheta) * npol, self.constant_tbdown, dtype=np.float32)

    def tbup(self, frequency, costheta, npol):
        if isinstance(self.constant_tbup, dict):
            return np.full(len(costheta) * npol, self.constant_tbup[frequency], dtype=np.float32)
        else:
            return np.full(len(costheta) * npol, self.constant_tbup, dtype=np.float32)

    def trans(self, frequency, costheta, npol):
        if isinstance(self.constant_trans, dict):
            return np.full(len(costheta) * npol, self.constant_trans[frequency], dtype=np.float32)
        else:
            return np.full(len(costheta) * npol, self.constant_trans, dtype=np.float32)
