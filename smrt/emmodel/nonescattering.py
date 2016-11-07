# coding: utf-8

"""Non-scattering medium can be applied to medium without heteoreneity (like water or pure ice).

"""

# Stdlib import

# other import
import numpy as np


# local import
from ..core.globalconstants import C_SPEED


class NoneScattering(object):
    """
    """
    def __init__(self, sensor, layer):

        self.frac_volume = layer.frac_volume
        self.e0 = layer.permittivity(0, sensor.frequency)  # background permittivity
        self.eps = layer.permittivity(1, sensor.frequency)  # scatterer permittivity
        # Wavenumber in free space
        self.k0 = 2 * np.pi * sensor.frequency / C_SPEED

        # General lossy medium under assumption of low-loss medium.
        self.ka = self.k0 * self.eps.imag / np.sqrt(self.eps.real)
        # no scattering
        self.ks = 0

    def basic_check(self):
        # Need to be defined
        pass

    def set_max_mode(self, m_max):
        """
        """
        self.m_max = m_max

    def ft_even_phase(self, m, mu):
        """ Non-scattering phase matrix.

            Returns : null phase matrix

        """

        npol = 2 if m == 0 else 3

        return np.zeros((npol * len(mu), npol * len(mu)))

    def phase(self, mu, phi):
        """Non-scattering phase matrix.

            Returns : null phase matrix

        """
        npol = 2
        return np.zeroes((npol * len(mu), npol * len(mu)))

    def ke(self, mu):
        return np.full(len(mu), self.ka)

    def effective_permittivity(self):
        return self.eps
