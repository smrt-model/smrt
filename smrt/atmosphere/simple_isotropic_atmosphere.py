# coding: utf-8
"""Implement an isotropic atmosphere with prescribed emission (up and down) and transmittivity

"""

# Stdlib import


# other import
import numpy as np

# local import
#from ..core.error import SMRTError


#
# For the developers:
# A valid Atmosphere is any object that provide a tbdown, tbup and trans functions function of frequency and costheta.
# costheta can be an array or a value, both must be handled.
# The implementation below is independent of the frequency and the propagation angle
#


class SimpleIsotropicAtmosphere(object):

    def __init__(self, tbdown=0, tbup=0, trans=1):

        self.constant_tbdown = tbdown
        self.constant_tbup = tbup
        self.constant_trans = trans

    def tbdown(self, frequency, costheta, npol):
        return np.full(len(costheta)*npol, self.constant_tbdown, dtype=np.float32)

    def tbup(self, frequency, costheta, npol):
        return np.full(len(costheta)*npol, self.constant_tbup, dtype=np.float32)

    def trans(self, frequency, costheta, npol):
        return np.full(len(costheta)*npol, self.constant_trans, dtype=np.float32)

