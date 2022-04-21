# coding: utf-8

"""Non-scattering medium can be applied to medium without heteoreneity (like water or pure ice layer).

"""

# Stdlib import

# other import
import numpy as np


# local import
from ..core.globalconstants import C_SPEED
from ..core.lib import smrt_matrix
from .common import extinction_matrix


class NonScattering(object):
    """
    """

    def __init__(self, sensor, layer):

        self.frac_volume = layer.frac_volume

        self.e0 = layer.permittivity(0, sensor.frequency)  # background permittivity
        self.eps = layer.permittivity(1, sensor.frequency)  # scatterer permittivity

        # Wavenumber in free space
        self.k0 = 2 * np.pi * sensor.frequency / C_SPEED

        self.ka = 2 * self.k0 * np.sqrt(self.effective_permittivity()).imag
        # no scattering
        self.ks = 0

    def basic_check(self):
        # Need to be defined
        pass

    def ft_even_phase(self, mu_s, mu_i, m_max, npol=None):
        """ Non-scattering phase matrix.

            Returns : null phase matrix

        """
        if npol is None:
            npol = 2 if m_max == 0 else 3

        return smrt_matrix.zeros((npol, npol, m_max + 1, len(mu_s), len(mu_i)))

    def phase(self, mu_s, mu_i, dphi, npol=2):
        """Non-scattering phase matrix.

            Returns : null phase matrix

        """

        return smrt_matrix.zeros((npol, npol, len(dphi), len(mu_s), len(mu_i)))

    def ke(self, mu, npol=2):
        return extinction_matrix(self.ka, mu=mu, npol=npol)

    def effective_permittivity(self):
        # very basic mixing formula. It is recommended to use either with frac_volume=0 or 1 a better mixings when available.
        return self.e0 * (1 - self.frac_volume) + self.eps * self.frac_volume
