"""
Provide the flat interface boundary condition between layers characterized by their effective permittivities. 

The reflection and transmission are computed using the Fresnel coefficient from :py:mod:`smrt.core.fresnel`.
"""

from smrt.core.lib import smrt_matrix
from smrt.core.fresnel import fresnel_reflection_matrix, fresnel_transmission_matrix
from smrt.core.interface import Interface


class Flat(Interface):
    """
    Implement a flat surface. 
    
    The reflection is in the specular direction and the coefficient is calculated with the Fresnel coefficients.
    """
    args = []
    optional_args = {}

    def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """
        Compute the reflection coefficients. 
        
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

        return fresnel_reflection_matrix(eps_1, eps_2, mu1, npol)

    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        return smrt_matrix(0)

    def coherent_transmission_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """
        Compute the transmission coefficients.
        
        Coefficients are calculated for an array of incidence angles (given by their cosine) in medium 1. Medium 2 is where the
        beam is transmitted.

        Args:
            frequency: Frequency of the incident wave.
            eps_1: Permittivity of the medium where the incident beam is propagating.
            eps_2: Permittivity of the other medium.
            mu1: Array of cosine of incident angles.
            npol: Number of polarization.

        Returns:
            The transmission matrix.
        """

        return fresnel_transmission_matrix(eps_1, eps_2, mu1, npol)

    def diffuse_transmission_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        return smrt_matrix(0)
