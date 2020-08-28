

"""
Implement the interface boundary condition under IEM formulation provided by Fung et al. 1992. in IEEE TGRS 
with an extended domain of validity (for large roughness or correlation length) by switching the Fresnel 
coefficients according to Brogioni et al. 2010, IJRS (doi: 10.1080/01431160903232808). A better but more
complex approach is given by Wu et al. 2003 (to be implemented).

Use of this code requires special attention.

"""

import numpy as np

from smrt.core.fresnel import fresnel_coefficients
from smrt.interface.iem_fung92 import IEM_Fung92
from smrt.core.error import SMRTError


class IEM_Fung92_Briogoni10(IEM_Fung92):
    """A moderate rough surface model with backscatter, specular reflection and transmission only. Use with care!

"""

    def check_validity(self, ks, kl, eps_r):

        # check validity
        if ks > 3:
            raise SMRTError("Warning, roughness_rms is too high for the given wavelength. Limit is ks < 3. Here ks=%g" % ks)

    def fresnel_coefficients(self, eps_1, eps_2, mu_i, ks, kl):
        """calculate the fresnel coefficients at the angle mu_i or 0° depending on ks*kl. The transition is abrupt."""

        if ks * kl > np.sqrt(eps_2 / eps_1):
            Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, 1)
        else:
            Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_i)
        return Rv, Rh
