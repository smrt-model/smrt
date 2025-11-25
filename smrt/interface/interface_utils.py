"""This modules provides common utility functions for the flat and rough interfaces in SMRT."""

import numpy as np

from smrt.core.fresnel import (
    fresnel_reflection_matrix,
    fresnel_transmission_matrix,
)
from smrt.core.globalconstants import C_SPEED
from smrt.core.lib import abs2


class KirchoffApproximationCoherentInterfaceMixin:
    """This mixin provides the coherent reflection and transmission matrices under the Kirchoff Approximation which is
    also found in SPM and IEM."""

    def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """
        Compute the specular reflection coefficients.

        Coefficients are calculated for an array of incidence angles (given by their cosine) in medium 1. Medium 2 is where the
        beam is transmitted.

        Args:
            frequency: Frequency of the incident wave.
            eps_1: Permittivity of the medium where the incident beam is propagating.
            eps_2: Permittivity of the other medium.
            mu1: Array of cosine of incident angles.
            npol: Number of polarization.

        Returns:
            The reflection matrix.
        """
        k2 = (2 * np.pi * frequency / C_SPEED) ** 2 * abs2(eps_1)
        # Eq: 2.1.94 in Tsang 2001 Tome I
        return fresnel_reflection_matrix(eps_1, eps_2, mu1, npol) * np.exp(-4 * k2 * self.roughness_rms**2 * mu1**2)

    def coherent_transmission_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """
        Compute the transmission coefficients.

        Coefficients are calculated for the azimuthal mode m and for an array of incidence angles (given by their cosine) in medium 1.
        Medium 2 is where the beam is transmitted.

        Args:
            frequency: Frequency of the incident wave.
            eps_1: Permittivity of the medium where the incident beam is propagating.
            eps_2: Permittivity of the other medium.
            mu1: Array of cosine of incident angles.
            npol: Number of polarization.

        Returns:
            The transmission matrix.
        """
        k0 = 2 * np.pi * frequency / C_SPEED

        k_iz = k0 * np.sqrt(eps_1).real * mu1
        k_sz = k0 * np.sqrt(eps_2 - (1 - mu1**2) * eps_1).real

        return fresnel_transmission_matrix(eps_1, eps_2, mu1, npol) * np.exp(
            -((k_sz - k_iz) ** 2) * self.roughness_rms**2
        )
