# coding: utf-8
"""Implement an isotropic atmosphere with prescribed frequency-dependent emission (up and down) and transmittance.

 TB and transmissivity can be specified as a constant, or a frequency-dependent dictionary

 To make an atmosphere, it is recommended to use the helper function :py:func:`~smrt.inputs.make_model.make_atmosphere`.


Examples::

    # the full path import is required
    from smrt import make_atmosphere

    # Constant
    atmos = make_atmosphere("simple_isotopic_atmosphere", tb_down=20., tb_up=6., transmittance=1)

    # Frequency-dependent
    atmos = make_atmosphere("simple_isotopic_atmosphere", tb_down={10e9: 15.2, 21e9: 23.5})

"""

# Stdlib import
from warnings import warn

# other import
import numpy as np

# local import
from ..core.atmosphere import AtmosphereBase, AtmosphereResult

#
# For the developers:
# A valid atmosphere is any object that provide a tbdown, tbup and trans functions function of frequency and costheta.
# costheta can be an array or a value, both must be handled.
# The implementation below is independent of the propagation angle
#


def make_atmosphere(tb_down=0, tb_up=0, transmittance=1):
    """ Construct an atmosphere instance.

    """
    warn("""This function 'make_atmosphere' is going to be depreciated. Use smrt.inputs.make_medium.make_atmosphere or the
short cut smrt.make_atmosphere instead.""", DeprecationWarning)

    # create the instance
    return SimpleIsotropicAtmosphere(tb_down=tb_down, tb_up=tb_up, transmittance=transmittance)


class SimpleIsotropicAtmosphere(AtmosphereBase):

    def __init__(self, tb_down=0., tb_up=0., transmittance=1.):

        self.constant_tbdown = tb_down
        self.constant_tbup = tb_up
        self.constant_trans = transmittance

    def run(self, frequency, costheta, npol):

        def create_array(x):
            if isinstance(x, dict):
                x = x[frequency]
            return np.full(len(costheta) * npol, x)

        return AtmosphereResult(
            tb_down=create_array(self.constant_tbdown),
            tb_up=create_array(self.constant_tbup),
            transmittance=create_array(self.constant_trans))
