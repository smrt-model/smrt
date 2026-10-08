"""This modules provides common utility functions for the flat and rough interfaces in SMRT."""

from abc import ABCMeta

import numpy as np

from smrt.core.fresnel import (
    field_fresnel_matrix,
    fresnel_reflection_matrix,
    fresnel_transmission_matrix,
)
from smrt.core.globalconstants import TWO_PI, WAVENUMBER_PER_HZ
from smrt.core.lib import abs2, cached_roots_legendre


class KirchoffApproximationCoherentInterfaceMixin(metaclass=ABCMeta):
    """This mixin provides the coherent reflection and transmission matrices under the Kirchoff Approximation which is
    also found in SPM and IEM.
    """

    def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """Compute the specular reflection coefficients.

        Coefficients are calculated for an array of incidence angles (given by their cosine) in medium 1. Medium 2 is
        where the beam is transmitted.

        Args:
            frequency: Frequency of the incident wave.
            eps_1: Permittivity of the medium where the incident beam is propagating.
            eps_2: Permittivity of the other medium.
            mu1: Array of cosine of incident angles.
            npol: Number of polarization.

        Returns:
            The reflection matrix.
        """

        ks = WAVENUMBER_PER_HZ * frequency * self.roughness_rms
        ks2 = ks**2 * abs2(eps_1)
        # Eq: 2.1.94 in Tsang 2001 Tome I
        r, _ = fresnel_reflection_matrix(eps_1, eps_2, mu1, npol)
        return r * np.exp(-4 * ks2 * mu1**2)

    def coherent_transmission_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """Compute the transmission coefficients.

        Coefficients are calculated for the azimuthal mode m and for an array of incidence angles (given by their
        cosine) in medium 1. Medium 2 is where the beam is transmitted.

        Args:
            frequency: Frequency of the incident wave.
            eps_1: Permittivity of the medium where the incident beam is propagating.
            eps_2: Permittivity of the other medium.
            mu1: Array of cosine of incident angles.
            npol: Number of polarization.

        Returns:
            The transmission matrix.
        """
        k0 = WAVENUMBER_PER_HZ * frequency

        t, mu2 = fresnel_transmission_matrix(eps_1, eps_2, mu1, npol)

        ks_iz = k0 * np.sqrt(eps_1).real * mu1 * self.roughness_rms
        ks_sz = k0 * np.sqrt(eps_2).real * mu2 * self.roughness_rms

        return t * np.exp(-((ks_sz - ks_iz) ** 2))

    def field_matrix(self, frequency, eps_1, eps_2, mu1):
        """Compute the specular reflection and transmission field coefficients.

        Coefficients are calculated for an array of incidence angles (given by their cosine) in medium 1. Medium 2 is
        where the beam is transmitted.

        Args:
            frequency: Frequency of the incident wave.
            eps_1: Permittivity of the medium where the incident beam is propagating.
            eps_2: Permittivity of the other medium.
            mu1: Array of cosine of incident angles.
            npol: Number of polarization.
        """
        # Eq: 2.1.94 in Tsang 2001 Tome I
        r, t, mu2 = field_fresnel_matrix(eps_1, eps_2, mu1)

        ks = WAVENUMBER_PER_HZ * frequency * self.roughness_rms
        ks_iz = ks * np.sqrt(eps_1).real * mu1
        ks_sz = ks * np.sqrt(eps_2).real * mu2

        return (
            r * np.exp(-2 * ks_iz**2),
            t * np.exp(-0.5 * ((ks_sz - ks_iz) ** 2)),
        )


class HemisphericalIntegrationMixin(metaclass=ABCMeta):
    """This mixin provides hemispherically integrated reflection and transmission coefficients.

    It can be used for emissivity calculation, if the rough surface theory conserve energy which is rarely the case.
    It can be used for energy conservation and debugging purpose also.

    """

    def reflection_coefficients(self, frequency, eps_1, eps_2, mu_i, n_mu=128, n_phi=128):
        # for debugging only at this stage

        mu, weights = cached_roots_legendre(n_mu, 0, 1)
        dphi = np.linspace(0, TWO_PI, n_phi, endpoint=False)

        R = self.diffuse_reflection_matrix(frequency, eps_1, eps_2, mu, mu_i, dphi, 2)

        # integrate the pola first, then the azimuth and last the mu
        R = R.values.sum(axis=(0, 2))
        return TWO_PI / n_phi * np.einsum("j...,ij...->i...", weights, R)

    def transmission_coefficients(self, frequency, eps_1, eps_2, mu_i, n_mu=128, n_phi=128):
        # for debugging only at this stage

        mu, weights = cached_roots_legendre(n_mu, 0, 1)
        dphi = np.linspace(0, TWO_PI, n_phi, endpoint=False)

        T = self.diffuse_transmission_matrix(frequency, eps_1, eps_2, mu, mu_i, dphi, 2)

        # integrate the pola first, then the azimuth and last the mu
        T = T.values.sum(axis=(0, 2))
        return TWO_PI / n_phi * np.einsum("j...,ij...->i...", weights, T)
