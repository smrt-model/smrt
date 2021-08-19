"""A transparent interface (no reflection). Useful for the unit-test mainly.

"""

from smrt.core.lib import smrt_matrix, len_atleast_1d


class Transparent(object):

    args = []
    optional_args = {}

    def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):

        """compute the reflection coefficients for the azimuthal mode m
           and for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium
        :param mhu1: array of cosine of incident angles
        :param npol: number of polarization

"""
        assert len(mu1.shape) == 1  # 1D array

        return smrt_matrix.zeros((npol, len(mu1)))

    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        return smrt_matrix(0)

    def coherent_transmission_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """compute the transmission coefficients for the azimuthal mode m
           and for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium
        :param mu1: array of cosine of incident angles
        :param npol: number of polarization

        :return: the transmission matrix
"""
        return smrt_matrix.ones((npol, len_atleast_1d(mu1)))

    def diffuse_transmission_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        return smrt_matrix(0)
