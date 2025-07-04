"""
Provide interface boundary condition under IEM formulation with an extended domain.

The interface boundary condition under IEM formulation provided by Fung et al. 1992 in IEEE TGRS with an extended domain of 
validity (for large roughness or correlation length) by switching the Fresnel coefficients according to Brogioni et al. 2010. A better 
but more complex approach is given by Wu et al. 2004 (to be implemented).

Note:
    Reflection and transmission matrix are the same as :py:mod:`~smrt.interface.iem_fung92.py`. Only change are the fresnel coefficients.

References:
    Brogioni, M., Pettinato, S., Macelloni, G., Paloscia, S., Pampaloni, P., Pierdicca, N., & Ticconi, F. (2010). Sensitivity of bistatic
    scattering to soil moisture and surface roughness of bare soils. International Journal of Remote Sensing, 31(15), 4227–4255. 
    https://doi.org/10.1080/01431160903232808

    Fung, A.K, Zongqian, L., and Chen, K.S. (1992). Backscattering from a randomly rough dielectric surface. IEEE TRANSACTIONS ON 
    GEOSCIENCE AND REMOTE SENSING, 30-2. https://doi.org/10.1109/36.134085

    Wu, T-D. and Chen, K-S. (2004). A reappraisal of the validity of the IEM model for backscattering from rough surfaces. IEEE Transactions
    on Geoscience and Remote Sensing, 42-4. https://doi.org/10.1109/TGRS.2003.815405
"""

import numpy as np

from smrt.core.fresnel import fresnel_coefficients
from smrt.interface.iem_fung92 import IEM_Fung92
from smrt.core.error import SMRTError


class IEM_Fung92_Briogoni10(IEM_Fung92):
    """
    Implement a moderate rough surface model with backscatter, specular reflection and transmission only. Use with care!

    Calculate the fresnel coefficients at the angle mu_i or 0° depending on ks*kl. The transition is abrupt. The fresnel coefficients 
    are computed with mu = 1 for ks * kl > np.sqrt(eps_2 / eps_1).
    """

    def check_validity(self, ks, kl, eps_r):

        # check validity
        if ks > 3:
            raise SMRTError("Warning, roughness_rms is too high for the given wavelength. Limit is ks < 3. Here ks=%g" % ks)

    def fresnel_coefficients(self, eps_1, eps_2, mu_i, ks, kl):
        # """
        # Calculate the fresnel coefficients at the angle mu_i or 0° depending on ks*kl. The transition is abrupt.
        # """

        if ks * kl > np.sqrt(eps_2 / eps_1):
            Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, 1)
        else:
            Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_i)
        return Rv, Rh
