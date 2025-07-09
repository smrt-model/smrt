"""
Implement a transparent interface (no reflection). Useful mainly for unit tests.
"""

from smrt.core.lib import smrt_matrix, len_atleast_1d


class Transparent(object):

    args = []
    optional_args = {}

    def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """
        Compute the reflection coefficients for the azimuthal mode m and for an array of incidence angles (given by their cosine) 
        in medium 1. Medium 2 is where the beam is transmitted.

        Args:
            frequency: Frequency of the incident wave.
            eps_1: Permittivity of the medium where the incident beam is propagating.
            eps_2: Permittivity of the other medium.
            mu1: Array of cosine of incident angles.
            npol: Number of polarization.

        Returns:
            The reflection matrix.
        """
        assert len(mu1.shape) == 1  # 1D array

        return smrt_matrix.zeros((npol, len(mu1)))

    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        return smrt_matrix(0)

    def coherent_transmission_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """
        Compute the transmission coefficients for the azimuthal mode m and for an array of incidence angles (given by their cosine) 
        in medium 1. Medium 2 is where the beam is transmitted.

        Args:
            frequency: Frequency of the incident wave.
            eps_1: Permittivity of the medium where the incident beam is propagating.
            eps_2: Permittivity of the other medium.
            mu1: Array of cosine of incident angles.
            npol: Number of polarization.

        Returns:
            The transmission matrix.
        """
        return smrt_matrix.ones((npol, len_atleast_1d(mu1)))

    def diffuse_transmission_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        return smrt_matrix(0)
