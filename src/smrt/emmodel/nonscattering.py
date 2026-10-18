# coding: utf-8

"""Non-scattering medium can be applied to medium without heterogeneity (like water or pure ice layer)."""

# Stdlib import

# other import
import numpy as np

# local import
from ..core.globalconstants import WAVENUMBER_PER_HZ
from ..core.lib import len_atleast_1d, smrt_matrix
from ..permittivity.generic_mixing_formula import polder_van_santen
from .emmodel_utils import IsotropicScatteringMixin, extinction_matrix


class NonScattering(IsotropicScatteringMixin):
    """ """

    def __init__(self, sensor, layer, *, precompute=True):
        self.sensor = sensor
        self.layer = layer
        self.frac_volume = layer.frac_volume
        self.k0 = WAVENUMBER_PER_HZ * sensor.frequency  # Wavenumber in free space

        if precompute:
            self.precompute()

    def precompute(self):
        freq = self.sensor.frequency
        layer = self.layer

        self.e0 = layer.permittivity(0, freq)  # background permittivity
        self.eps = layer.permittivity(1, freq)  # scatterer permittivity

        # effective permittivity using the classical polder van santen
        self.effective_permittivity = polder_van_santen(self.frac_volume, self.e0, self.eps)

        self.ka = 2 * self.k0 * np.sqrt(self.effective_permittivity).imag
        # no scattering
        self._ks = 0
        return self

    def basic_check(self):
        # Need to be defined
        pass

    def ft_even_phase(self, mu_s, mu_i, m_max, npol=None):
        """Non-scattering phase matrix.

        Args:
            mu_s:
            mu_i:
            m_max:
            npol:  (Default value = None)

        Returns:
            null phase matrix
        """
        if npol is None:
            npol = 2 if m_max == 0 else 3

        return smrt_matrix.zeros((npol, npol, m_max + 1, len_atleast_1d(mu_s), len_atleast_1d(mu_i)))

    def phase(self, mu_s, mu_i, dphi, npol=2):
        """Non-scattering phase matrix.

        Args:
            mu_s:
            mu_i:
            dphi:
            npol:  (Default value = 2)

        Returns:
            null phase matrix
        """
        return smrt_matrix.zeros(
            (
                npol,
                npol,
                len_atleast_1d(dphi),
                len_atleast_1d(mu_s),
                len_atleast_1d(mu_i),
            )
        )

    def ke(self, mu, npol=2):
        return extinction_matrix(self.ka, mu=mu, npol=npol)
