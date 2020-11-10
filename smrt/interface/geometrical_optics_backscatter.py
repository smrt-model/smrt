

"""
Implement the interface boundary condition under the Geometrical Approximation between layers charcterized by their effective permittivities. This code is
for backscatter only, that is, to use as a substrate and at low frequency when the backscatter is the main mecahnism, and conversely when mulitple scattering
and double bounce between snow and substrate are negligible. In any other case, it is recommended to use :py:class:`~smrt.interface.geometrical_optics.GeometricalOptics`.

The transmitted energy is also computed in an approximate way suitable for 1st order scattering. We use energy conservation to compute the total transmitted energy
and consider that all this energy is transmitted in the refraction (specular) direction.
"""

import numpy as np

from smrt.core.fresnel import fresnel_transmission_matrix, fresnel_coefficients  # a modifier quand on fusionne
from smrt.core.lib import smrt_matrix, len_atleast_1d
from smrt.core.interface import Interface
from smrt.interface.geometrical_optics import shadow_function, GeometricalOptics


class GeometricalOpticsBackscatter(Interface):
    """A very rough surface.

"""
    args = ["mean_square_slope"]
    optional_args = {"shadow_correction": True}


    def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """compute the reflection coefficients for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium
        :param mu1: array of cosine of incident angles
        :param npol: number of polarization

        :return: the reflection matrix
"""

        return smrt_matrix(0)


    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        """compute the reflection coefficients for an array of incident, scattered and azimuth angles 
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium
        :param mu1: array of cosine of incident angles
        :param npol: number of polarization

        :return: the reflection matrix
"""
        mu_s = np.atleast_1d(mu_s)
        mu_i = np.atleast_1d(mu_i)

        if not np.allclose(mu_s, mu_i) or not np.allclose(dphi, np.pi):
            raise NotImplementedError("Only the backscattering coefficient is implemented at this stage. This is a very preliminary implementation")

        if len(np.atleast_1d(dphi)) != 1:
            raise NotImplementedError("Only the backscattering coefficient is implemented at this stage. ")

        R_normal, _, _ = fresnel_coefficients(eps_1, eps_2, np.ones(1))

        tantheta_i2 = 1 / mu_i ** 2 - 1

        smrt_norm = 1 / (4 * np.pi)

        gamma = smrt_norm / (2 * self.mean_square_slope) * np.abs(R_normal)**2 / mu_i**5 * np.exp(- tantheta_i2 / (2 * self.mean_square_slope))

        if self.shadow_correction:
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
                coef = 0.5
            elif (m % 2) == 1:
                coef = -1.0
            else:
                coef = 1.0

            coef /= m_max + 0.5  # ad hoc normalization to get the right backscatter. This is a trick to deal with the dirac.

            diffuse_refl_coeff[:, m, :] = coef * gamma[:, :]

        return diffuse_refl_coeff

    def coherent_transmission_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """compute the transmission coefficients for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted. While Geometrical Optics, we here consider that power not reflected
           is scattered in the specular transmitted direction. This is an approximation which is reasonable in the context of a "1st order"
           geometrical optics.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium
        :param mu1: array of cosine of incident angles
        :param npol: number of polarization

        :return: the transmission matrix
"""
        go = GeometricalOptics(mean_square_slope=self.mean_square_slope, shadow_function=self.shadow_correction)
        total_reflection = go.reflection_coefficients(frequency, eps_1, eps_2, mu1)

        transmission_matrix = smrt_matrix.zeros((npol, len_atleast_1d(mu1)))
        transmission_matrix[0] = 1 - total_reflection[0]
        transmission_matrix[1] = 1 - total_reflection[1]

        return transmission_matrix



