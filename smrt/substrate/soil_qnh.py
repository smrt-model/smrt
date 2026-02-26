# coding: utf-8

"""
Implements the QNH soil model proposed by Wang, 1983.

This model is for the passive mode and is not suitable for the active mode.

Args:
    Q: Optional; default is 0.
    N: Optional; default is 0.
    Nv: Optional; default is NaN.
    Nh: Optional; default is NaN.
    H: Required; the roughness rms (note: called roughness_rms in soil_wegmuller).

Examples:
    ```python
    soil = make_soil("soil_qnh", "dobson85_peplinski95", moisture=0.2, sand=0.4, clay=0.3, drymatter=1100,
                     Q=0, N=0, H=1e-2)
    ```
"""

import numpy as np

from smrt.core.fresnel import fresnel_reflection_matrix, fresnel_transmission_matrix

# local import
from smrt.core.interface import Substrate


class SoilQNH(Substrate):
    args = ["H"]
    optional_args = {"Q": 0.0, "N": 0.0, "Nv": np.nan, "Nh": np.nan}

    def adjust(self, rh, rv, mu1):
        # in place modification of rh and rv for the rough soil QNH reflectivity model

        if np.isnan(self.Nv):
            self.Nv = self.N
        if np.isnan(self.Nh):
            self.Nh = self.N

        coef_h = np.exp(-self.H * (mu1**self.Nh))
        coef_v = np.exp(-self.H * (mu1**self.Nv))

        trv = (
            (1 - self.Q) * rv + self.Q * rh
        ) * coef_v  # trv is temporary because rv (which is a view) is needed in the next line.
        rh[:] = ((1 - self.Q) * rh + self.Q * rv) * coef_h
        rv[:] = trv  # copy back to the view.

    def specular_reflection_matrix(self, frequency, eps_1, mu1, npol):
        eps_2 = self.permittivity(frequency)

        reflection_coefficients = fresnel_reflection_matrix(eps_1, eps_2, mu1, npol)

        self.adjust(reflection_coefficients[1], reflection_coefficients[0], mu1)

        if npol >= 3:
            # don't modify the third compoment... this is an approximation, as the third component should be affected by the roughness...
            # don't use this model for active mode
            pass
        if npol == 4:
            raise NotImplementedError("to be implemented, the matrix is not diagonal anymore")

        return reflection_coefficients

    def emissivity_matrix(self, frequency, eps_1, mu1, npol):
        # this function is a bit complex because we have to change first and second component but not the third one.
        # this is an approximation, as the third component should be affected by the roughness...

        eps_2 = self.permittivity(frequency)

        transmission_coefficients = fresnel_transmission_matrix(eps_1, eps_2, mu1, npol)

        rh = 1 - transmission_coefficients[1]
        rv = 1 - transmission_coefficients[0]

        self.adjust(rh, rv, mu1)

        transmission_coefficients[1] = 1 - rh  # back to transmission coefficients
        transmission_coefficients[0] = 1 - rv

        if npol >= 3:
            # don't modify the third compoment... don't know what to do with it !
            pass
        if npol == 4:
            raise Exception("to be implemented, the matrix is not diagonal anymore")

        return transmission_coefficients
