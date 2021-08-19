

"""
Implement the flat interface boundary condition between layers charcterized by their effective permittivities. The reflection and transmission
are computed using the Fresnel coefficient.

"""

from smrt.core.lib import smrt_matrix
from smrt.core.fresnel import fresnel_reflection_matrix, fresnel_transmission_matrix
from smrt.core.interface import Interface


class Flat(Interface):
    """A flat surface. The reflection is in the specular direction and the coefficient is calculated with the Fresnel coefficients

"""
    args = []
    optional_args = {}

    def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """compute the reflection coefficients for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium.
        :param mu1: array of cosine of incident angles.
        :param npol: number of polarization.

        :return: the reflection matrix
"""

        return fresnel_reflection_matrix(eps_1, eps_2, mu1, npol)

    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        return smrt_matrix(0)

    def coherent_transmission_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """compute the transmission coefficients for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium.
        :param mu1: array of cosine of incident angles.
        :param npol: number of polarization.

        :return: the transmission matrix
"""

        return fresnel_transmission_matrix(eps_1, eps_2, mu1, npol)

    def diffuse_transmission_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        return smrt_matrix(0)
