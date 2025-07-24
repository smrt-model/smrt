"""
Provide the interface boundary condition under the Geometrical Approximation between layers characterized by their
effective permittivities. 

This code is for backscatter only, that is, to use as a substrate and at low frequency when
the backscatter is the main mecahnism, and conversely when mulitple scattering and double bounce between snow and
substrate are negligible. In other case, it is recommended to use :py:mod:`~smrt.interface.geometrical_optics`.

Note:
    The transmitted energy is also computed in an approximate way suitable for first order scattering such as 
    :py:mod:`smrt.rtsolver.nadir_lrm_altimetry`. It uses energy conservation to compute the total transmitted energy and considers that 
    all this energy is transmitted in the  refracted direction. This approach compensates for the deficiencies of first order scattering
    RT solvers.
"""

import numpy as np

from smrt.core.fresnel import fresnel_transmission_matrix, fresnel_coefficients  # a modifier quand on fusionne
from smrt.core.lib import smrt_matrix, len_atleast_1d
from smrt.core.interface import Interface
from smrt.interface.geometrical_optics import shadow_function, GeometricalOptics


class GeometricalOpticsBackscatter(GeometricalOptics):
    """
    Implement a very rough surface for backscatter.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        return smrt_matrix(0)

    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        """
        Compute the diffuse reflection coefficients.

        Coefficients are calculated for an array of incidence angles (given by their cosine) in medium 1. Medium 2 is where the 
        beam is transmitted.

        Args:
            frequency: Frequency of the incident wave.
            eps_1: Permittivity of the medium where the incident beam is propagating.
            eps_2: Permittivity of the other medium.
            mu_s: Array of cosine of scattered angles.
            mu_i: Array of cosine of incident angles.
            dphi: Azimuth angles.
            npol: Number of polarization.

        Returns:
            The reflection matrix.
        """
        mu_s = np.atleast_1d(mu_s)
        mu_i = np.atleast_1d(mu_i)

        if not np.allclose(mu_s, mu_i) or not np.allclose(dphi, np.pi):
            raise NotImplementedError(
                "Only the backscattering coefficient is implemented at this stage. This is a very preliminary implementation"
            )

        if len(np.atleast_1d(dphi)) != 1:
            raise NotImplementedError("Only the backscattering coefficient is implemented at this stage. ")

        R_normal, _, _ = fresnel_coefficients(eps_1, eps_2, np.ones(1))

        tantheta_i2 = 1 / mu_i**2 - 1

        smrt_norm = 1 / (4 * np.pi)

        gamma = (
            smrt_norm
            / (2 * self.mean_square_slope)
            * np.abs(R_normal) ** 2
            / mu_i**5
            * np.exp(-tantheta_i2 / (2 * self.mean_square_slope))
        )

        if self.shadow_correction:
            with np.errstate(divide="ignore"):
                gamma *= 1 / (1 + shadow_function(self.mean_square_slope, 1 / np.sqrt(tantheta_i2)))

        reflection_coefficients = smrt_matrix.zeros((npol, len(mu_i)))
        reflection_coefficients[0] = gamma
        reflection_coefficients[1] = gamma

        return reflection_coefficients

    def ft_even_diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol):
        assert mu_s is mu_i

        diffuse_refl_coeff = smrt_matrix.zeros((npol, m_max + 1, len(mu_i)))

        gamma = self.diffuse_reflection_matrix(frequency, eps_1, eps_2, mu_s, mu_i, dphi=np.pi, npol=npol)

        for m in range(m_max + 1):
            if m == 0:
                coef = 1
            elif (m % 2) == 1:
                coef = -2.0
            else:
                coef = 2.0

            coef /= 1 + 2 * m_max  # this normalization is used to spread the energy in the backscatter over all modes

            diffuse_refl_coeff[:, m, :] = coef * gamma[:, :]

        return diffuse_refl_coeff

    def coherent_transmission_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """
        Compute the coherent transmission coefficients.

        Coefficients are calculated for an array of incidence angles (given by their cosine) in medium 1. Medium 2 is where the 
        beam is transmitted. While Geometrical Optics, it here considers that power not reflected is scattered in the specular 
        transmitted direction. This is an approximation which is reasonable in the context of a "1st order" geometrical optics.

        Args:
            frequency: Frequency of the incident wave.
            eps_1: Permittivity of the medium where the incident beam is propagating.
            eps_2: Permittivity of the other medium.
            mu1: Array of cosine of incident angles.
            npol: Number of polarization.

        Returns:
            The transmission matrix.
        """
        go = GeometricalOptics(mean_square_slope=self.mean_square_slope, shadow_function=self.shadow_correction)
        total_reflection = go.reflection_coefficients(frequency, eps_1, eps_2, mu1)

        transmission_matrix = smrt_matrix.zeros((npol, len_atleast_1d(mu1)))
        transmission_matrix[0] = 1 - total_reflection[0]
        transmission_matrix[1] = 1 - total_reflection[1]

        return transmission_matrix
