# coding: utf-8
"""
Implements a one-layer atmosphere with prescribed frequency-dependent emission (up and down) and transmittance.

TB and transmissivity are given as array of incidence and optionally as frequency-dependent dictionary.

In the current implementation, there are two constraints:
- All the parameters (theta, tbub, tbdown and transmission) must be 1D list or arrays of the same length.
- The provided theta angles must cover the widest range possible in [0°, 90°] given that only interpolation is implemented at this stage.
  When the RT solver calls this atmosphere, the requested cosines must be within the range of provided theta values.
  Ideally 0° and 90° should be provided.

 To make an atmosphere, it is recommended to use the helper function :py:func:`~smrt.inputs.make_model.make_atmosphere`.


Examples::

    # the full path import is required
    from smrt import make_atmosphere

    # Incident dependent only
    atmos = make_atmosphere(
        "simple_atmosphere",
        theta=[0, 40, 89],
        tb_down=[20.0, 25, 40],
        tb_up=[18.0, 23, 38],
        transmittance=[0.95, 0.90, 0.80],
    )

    # Frequency-dependent
    atmos = make_atmosphere(
        "simple_atmosphere",
        theta=[10, 40, 90],
        tb_down={37e9: [20.0, 25, 40]},
        tb_up={37e9: [18.0, 23, 38]},
        transmittance={37e9: [0.95, 0.90, 0.80]},
)

"""

# other import
import numpy as np

from smrt.core.error import SMRTError

# local import
from ..core.atmosphere import AtmosphereBase, AtmosphereResult


class SimpleAtmosphere(AtmosphereBase):
    def __init__(self, theta, tb_down, tb_up, transmittance):
        if len(theta) < 2:
            raise SMRTError(
                "The theta parameter must be a list or array of angles in degrees with at least two values (0° and close to 90° recommended)."
            )

        costheta = np.cos(np.deg2rad(theta))

        # sort by increasing costheta
        i = np.argsort(costheta)

        self.theta = np.array(theta)[i]
        self.costheta = np.array(costheta)[i]
        self.tb_down = _sort_variable(tb_down, i, "tb_down", len(self.theta))
        self.tb_up = _sort_variable(tb_up, i, "tb_up", len(self.theta))
        self.transmittance = _sort_variable(transmittance, i, "transmittance", len(self.theta))

    def run(self, frequency, costheta, npol):
        def interpolate(x):
            if isinstance(x, dict):
                if frequency not in x.keys():
                    raise SMRTError(f"Frequency {frequency} not defined in atmosphere.")
                else:
                    x = x[frequency]
            x = np.interp(costheta, self.costheta, x)
            return np.stack([x] * npol)

        return AtmosphereResult(
            tb_down=interpolate(self.tb_down),
            tb_up=interpolate(self.tb_up),
            transmittance=interpolate(self.transmittance),
        )


def _sort_variable(x, sorted_index, name, length):
    if isinstance(x, dict):
        try:
            x = {key: np.array(x[key])[sorted_index] for key in x}
        except IndexError:
            raise SMRTError(
                "The length of the tb_down values must match the length of the theta array. "
                f"Got {len(name)} values for {length} angles."
            )
    else:
        x = np.array(x)[sorted_index]
    return x
