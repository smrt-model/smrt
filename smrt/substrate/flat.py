# coding: utf-8


"""
Implement the flat interface boundary for the bottom layer (substrate). The reflection and transmission
are computed using the Fresnel coefficients. This model does not take any specific parameter.

"""

# local import
from smrt.core.substrate import Substrate
from smrt.core.fresnel import fresnel_reflection_matrix, fresnel_transmission_matrix


class Flat(Substrate):

    args = []
    optional_args = {}

    def specular_reflection_matrix(self, frequency, eps_1, mu1, npol, compute_coherent_only):
        """compute the specular reflection coefficients for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted.

        :param npol: number of polarization
        :param frequency: frequency
        :param eps_1: permittivity of the medium where the incident beam is coming from.
        :param mu1: array of cosine of incident angles

        :return: the reflection matrix
"""
        eps_2 = self.permittivity(frequency)

        return fresnel_reflection_matrix(eps_1, eps_2, mu1, npol)

    def absorption_matrix(self, frequency, eps_1, mu1, npol, compute_coherent_only):
        """compute the absorption coefficients for an array of incidence angles (given by their cosine)
           in medium 1.

        :param npol: number of polarization
        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param mu1: array of cosine of incident angles

        :return: the transmission matrix
"""
        eps_2 = self.permittivity(frequency)

        return fresnel_transmission_matrix(eps_1, eps_2, mu1, npol)
