# coding: utf-8
"""Implement a one-layer atmosphere with prescribed frequency-dependent emission (up and down) and transmittance.

 TB and transmissivity are given as array of incidence and optional as frequency-dependent dictionnary.

 In the current implementation, there are two constraints:
 - All the parametres (theta, tbub, tbdown and transmission) must be 1D list or arrays of the same length
 - the provided theta angles must cover the widest range possible in [0°, 90°] given that only interpolation is implemented at this stage.
    When the RT solver calls this atmosphere, the requested cosines must be within the range of provided theta values.
    Ideally 0° and 90° should be provided.

 To make an atmosphere, it is recommended to use the helper function :py:func:`~smrt.inputs.make_model.make_atmosphere`.


Examples::

    # the full path import is required
    from smrt import make_atmosphere

    # Incident dependent only
    atmos = make_atmosphere("simple_atmosphere", theta=[0, 40, 89], tb_down=[20., 25, 40], tb_up=[18., 23, 38],
    transmittance=[0.95, 0.90, 0.80])

    # Frequency-dependent
    atmos = make_atmosphere("simple_atmosphere", theta={37e9: [10, 40, 90]}, tb_down={37e9: [20., 25, 40]}, tb_up={37e9: [18., 23, 38]})

"""


# other import
import numpy as np

from smrt.core.error import SMRTError

# local import
from ..core.atmosphere import AtmosphereBase, AtmosphereResult


class SimpleAtmosphere(AtmosphereBase):

    def __init__(self, theta, tb_down, tb_up, transmittance):

        if len(theta) < 2:
            raise SMRTError("theta must contains at least two values (0° and close to 90° recommended).")

        costheta = np.cos(np.deg2rad(theta))

        # sort by increasing costheta
        i = np.argsort(costheta)

        self.theta = np.array(theta)[i]
        self.costheta = np.array(costheta)[i]
        self.tbdown = np.array(tb_down)[i]
        self.tbup = np.array(tb_up)[i]
        self.trans = np.array(transmittance)[i]

    def run(self, frequency, costheta, npol):

        def interpolate(x):
            if isinstance(x, dict):
                x = x[frequency]
            return np.repeat(np.interp(costheta, self.costheta, x), npol)

        return AtmosphereResult(
            tb_down=interpolate(self.tbdown),
            tb_up=interpolate(self.tbup),
            transmittance=interpolate(self.trans))
