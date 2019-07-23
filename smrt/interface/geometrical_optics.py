

"""
Implement the interface boundary condition under the Geometrical Approximation between layers charcterized by their effective permittivities.
This approximation is suitable for surface with roughness much larger than the roughness scales, typically k*s >> 1 and k*l >> 1, where s the rms heigth and l 
the correlation length. The precise validity range must be investigated by the user, this code does not raise any warning. An important charcateristic of 
this approximation is that the scattering do not directly depend on frequency, the only (probably weak) dependence is through the permittivities of the media.

The model is parameterized by the mean_square_slope which can be calculated as mean_square_slope = 2*s**2/l**2 for surface with a Gaussian autocorrelation function.
Other equations may exist for other autocorrelation function.

This implementation is largely based on Tsang and Kong, Scattering of Electromagnetic Waves: Advanced Topics, 2001 (Tsang_tomeIII in the following)

"""

import numpy as np
import scipy.special

from smrt.core.fresnel import fresnel_transmission_matrix, fresnel_coefficients  # a modifier quand on fusionne
from smrt.core.lib import smrt_matrix, abs2, generic_ft_even_matrix
from smrt.interface.vector3 import vector3
from smrt.core.interface import Interface


class GeometricalOptics(Interface):
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
        mu_i = np.atleast_1d(mu_i)[np.newaxis, np.newaxis, :]
        mu_s = np.atleast_1d(mu_s)[np.newaxis, :, np.newaxis]
        dphi = np.atleast_1d(dphi)[:, np.newaxis, np.newaxis]

        sin_i = np.sqrt(1 - mu_i**2)
        sin_s = np.sqrt(1 - mu_s**2)

        cos_phi = np.cos(dphi)
        sin_phi = np.sin(dphi)

        ki = vector3.from_xyz(sin_i, 0, -mu_i)
        ks = vector3.from_xyz(sin_s * cos_phi, sin_s * sin_phi, mu_s)

        # compute the local reflection Fresnel coefficient
        kd = ki - ks   # in principe: *sqrt(eps_1), but in the following it appears everywhere as a ratio

        n = kd / (np.sign(kd.z) * kd.norm)  # EQ 2.1.223 #equivalent to np.vector3(kd_x / kd_z, kd_y / kd_z, 1)

        mu_local = -vector3.dot(n, ki)

        Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_local)
 
        # define polarizations
        hs = vector3.from_xyz(-sin_phi, cos_phi, 0)
        vs = vector3.from_xyz(mu_s * cos_phi, mu_s * sin_phi, -sin_s)

        hi = vector3.from_xyz(0, 1, 0)
        vi = vector3.from_xyz(-mu_i, 0, -sin_i)

        # compute the dot products
        hs_ki = vector3.dot(hs, ki)
        vs_ki = vector3.dot(vs, ki)

        hi_ks = vector3.dot(hi, ks)
        vi_ks = vector3.dot(vi, ks)

        fvv = abs2( hs_ki * hi_ks * Rh + vs_ki* vi_ks * Rv)   # Eqs 2.1.122 in Tsang_tomeIII
        fhh = abs2( vs_ki * vi_ks * Rh + hs_ki* hi_ks * Rv)

        fhv = abs2( vs_ki * hi_ks * Rh - hs_ki* vi_ks * Rv)
        fvh = abs2( hs_ki * vi_ks * Rh + vs_ki* hi_ks * Rv)
        
        reflection_coefficients = smrt_matrix.zeros((npol, npol, dphi.shape[0], mu_s.shape[1], mu_i.shape[2]))
        reflection_coefficients[0, 0] = fvv
        reflection_coefficients[0, 1] = fvh
        reflection_coefficients[1, 0] = fhv
        reflection_coefficients[1, 1] = fhh

        smrt_norm = 1 / (4 * np.pi)  #  divide by 4*pi because this is the norm for SMRT

        coef = smrt_norm / (2 * self.mean_square_slope) / mu_i * kd.norm2**2 / (vector3.cross(ki, ks).norm2**2 * kd.z**4) * \
                         np.exp( - (kd.x**2 + kd.y**2)/ (2 * kd.z**2 * self.mean_square_slope))  # Eq. 2.1.124 

        if self.shadow_correction:

            backward = dphi == np.pi
            higher_thetas = mu_s <= mu_i
            zero_i = backward & higher_thetas
            zero_s = backward & ~higher_thetas

            s = 1 / (1 + zero_i * shadow_function(self.mean_square_slope, mu_i/sin_i) + zero_s * shadow_function(self.mean_square_slope, mu_s/sin_s))
            coef *= s

        return reflection_coefficients * coef

    def ft_even_diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol):

        def reflection_function(dphi):
            return self.diffuse_reflection_matrix(frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol=npol)

        return generic_ft_even_matrix(reflection_function, m_max)

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

        return smrt_matrix(0)

    def diffuse_transmission_matrix(self, frequency, eps_1, eps_2, mu_t, mu_i, dphi, npol):
        """compute the transmission coefficients for an array of incident, scattered and azimuth angles 
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium
        :param mu_i: array of cosine of incident angles
        :param mu_t: array of cosine of transmitted wave angles
        :param npol: number of polarization

        :return: the transmission matrix
"""
        n_2 = np.sqrt(eps_2)
        n_1 = np.sqrt(eps_1)

        eta1_eta = n_1 / n_2  # eta1 is the impedance in medium 2 and eta in medium 1. Impedance is sqrt(permeability/permittivity)

        mu_i = np.atleast_1d(mu_i)[np.newaxis, np.newaxis, :]
        mu_t = np.atleast_1d(mu_t)[np.newaxis, :, np.newaxis]
        dphi = np.atleast_1d(dphi)[:, np.newaxis, np.newaxis]

        sin_i = np.sqrt(1 - mu_i**2)
        sin_t = np.sqrt(1 - mu_t**2)

        cos_phi = np.cos(dphi)
        sin_phi = np.sin(dphi)

        ki = vector3.from_xyz(sin_i, 0, -mu_i)
        kt = vector3.from_xyz(sin_t * cos_phi, sin_t * sin_phi, -mu_t)

        # compute the local transmission Fresnel coefficient
        ktd = n_1.real * ki - n_2.real * kt   # Eq 2.1.87

        n = ktd / (np.sign(ktd.z) * ktd.norm)  # Eq 2.1.128

        mu_local = -vector3.dot(n, ki)

        Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_local)

        n_kt = vector3.dot(n, kt)
        n_kt[mu_local < 0] = 0
 
        # define polarizations
        ht = vector3.from_xyz(-sin_phi, cos_phi, 0)
        vt = vector3.from_xyz(-mu_t * cos_phi, -mu_t * sin_phi, -sin_t)

        hi = vector3.from_xyz(0, 1, 0)
        vi = vector3.from_xyz(-mu_i, 0, -sin_i)

        # compute the cosines

        ht_ki = vector3.dot(ht, ki)
        vt_ki = vector3.dot(vt, ki)

        hi_kt = vector3.dot(hi, kt)
        vi_kt = vector3.dot(vi, kt)

        Wvv = abs2( ht_ki * hi_kt * (1 + Rh) + vt_ki * vi_kt * (1 + Rv) * eta1_eta)   # Eqs 2.1.130 in Tsang_tomeIII
        Whh = abs2( vt_ki * vi_kt * (1 + Rh) + ht_ki * hi_kt * (1 + Rv) * eta1_eta)

        Whv = abs2(-vt_ki * hi_kt * (1 + Rh) + ht_ki * vi_kt * (1 + Rv) * eta1_eta)
        Wvh = abs2( ht_ki * vi_kt * (1 + Rh) - vt_ki * hi_kt * (1 + Rv) * eta1_eta)
        
        transmission_coefficients = smrt_matrix.zeros((npol, npol, dphi.shape[0], mu_t.shape[1], mu_i.shape[2]))

        transmission_coefficients[0, 0] = Wvv
        transmission_coefficients[0, 1] = Wvh
        transmission_coefficients[1, 0] = Whv
        transmission_coefficients[1, 1] = Whh

        smrt_norm = 1 / (4 * np.pi)   # SMRT requires scattering coefficient / 4 * pi

        coef = smrt_norm * 2 * eps_2 / (eta1_eta * self.mean_square_slope) / mu_i * \
                ktd.norm2 * n_kt**2 / (vector3.cross(ki, kt).norm2**2 * ktd.z**4) * \
                np.exp(-(ktd.x**2 + ktd.y**2) / (2 * ktd.z**2 * self.mean_square_slope))   # Eq. 2.1.130   NB: k1^2 -> eps_2

        if self.shadow_correction:
            s = 1 / (1 + shadow_function(self.mean_square_slope, mu_i/sin_i) + shadow_function(self.mean_square_slope, mu_t/sin_t))
            coef *= s

        return transmission_coefficients * coef.real

    def reflection_integrand_for_energy_conservation_test(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        """function relevant to compute energy conservation. See p87 in Tsang_tomeIII.
"""
        mu_i = np.atleast_1d(mu_i)[np.newaxis, np.newaxis, :]
        mu_s = np.atleast_1d(mu_s)[np.newaxis, :, np.newaxis]
        dphi = np.atleast_1d(dphi)[:, np.newaxis, np.newaxis]

        sin_i = np.sqrt(1 - mu_i**2)
        sin_s = np.sqrt(1 - mu_s**2)

        cos_phi = np.cos(dphi)
        sin_phi = np.sin(dphi)

        ki = vector3.from_xyz(sin_i, 0, -mu_i)
        ks = vector3.from_xyz(sin_s * cos_phi, sin_s * sin_phi, mu_s)

        # compute the local reflection Fresnel coefficient
        kd = ki - ks   # in principe: *sqrt(eps_1), but in the following it appears everywhere as a ratio

        n = kd / (np.sign(kd.z) * kd.norm)  # EQ 2.1.223 #equivalent to np.vector3(kd_x / kd_z, kd_y / kd_z, 1)

        mu_local = -vector3.dot(n, ki)

        Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_local)
 
        # define polarizations
        hs = vector3.from_xyz(-sin_phi, cos_phi, 0)
        vs = vector3.from_xyz(mu_s * cos_phi, mu_s * sin_phi, -sin_s)

        #print(vector3.cross(vs, hs), ks)

        hi = vector3.from_xyz(0, 1, 0)
        vi = vector3.from_xyz(-mu_i, 0, -sin_i)

        #print(vector3.cross(vi, hi), ki)

        hs_ki = vector3.dot(hs, ki)
        vs_ki = vector3.dot(vs, ki)

        hi_ks = vector3.dot(hi, ks)
        vi_ks = vector3.dot(vi, ks)

        coef = 1/(2 * np.pi * self.mean_square_slope) * kd.norm2**2 / (4 * mu_i * vector3.cross(ki, ks).norm2 * kd.z**4) * \
                         np.exp( - (kd.x**2 + kd.y**2)/ (2 * kd.z**2 * self.mean_square_slope))  # Eq. 2.1.124 

        return coef * (hi_ks**2 * abs2(Rh) + vi_ks**2 * abs2(Rv)), \
               coef * (vi_ks**2 * abs2(Rh) + hi_ks**2 * abs2(Rv)) 


    def transmission_integrand_for_energy_conservation_test(self, frequency, eps_1, eps_2, mu_t, mu_i, dphi, npol):
        """function relevant to compute energy conservation. See p87 in Tsang_tomeIII.
"""
        n_2 = np.sqrt(eps_2)
        n_1 = np.sqrt(eps_1)

        mu_i = np.atleast_1d(mu_i)[np.newaxis, np.newaxis, :]
        mu_t = np.atleast_1d(mu_t)[np.newaxis, :, np.newaxis]
        dphi = np.atleast_1d(dphi)[:, np.newaxis, np.newaxis]

        sin_i = np.sqrt(1 - mu_i**2)
        sin_t = np.sqrt(1 - mu_t**2)

        cos_phi = np.cos(dphi)
        sin_phi = np.sin(dphi)

        ki = vector3.from_xyz(sin_i, 0, -mu_i)
        kt = vector3.from_xyz(sin_t * cos_phi, sin_t * sin_phi, -mu_t)

        # compute the local transmission Fresnel coefficient
        ktd = n_1.real * ki - n_2.real * kt   # Eq 2.1.87

        n = ktd / (np.sign(ktd.z) * ktd.norm)  # Eq 2.1.128

        mu_local = -vector3.dot(n, ki)

        Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_local)
 
        # define polarizations
        ht = vector3.from_xyz(-sin_phi, cos_phi, 0)
        vt = vector3.from_xyz(-mu_t * cos_phi, -mu_t * sin_phi, -sin_t)

        #print(vector3.cross(vt, ht), kt)

        hi = vector3.from_xyz(0, 1, 0)
        vi = vector3.from_xyz(-mu_i, 0, -sin_i)

        #print(vector3.cross(vi, hi), ki)

        ht_ki = vector3.dot(ht, ki)
        vt_ki = vector3.dot(vt, ki)

        hi_kt = vector3.dot(hi, kt)
        vi_kt = vector3.dot(vi, kt)

        Tv =  (hi_kt**2) * (1-abs2(Rh)) + (vi_kt**2) * (1-abs2(Rv))
        Th =  (vi_kt**2) * (1-abs2(Rh)) + (hi_kt**2) * (1-abs2(Rv))

        coef = 1/(2 * np.pi * self.mean_square_slope) * eps_2 * ktd.norm2 * vector3.dot(n, kt) * vector3.dot(n, ki)  \
                / (mu_i * vector3.cross(ki, kt).norm2 * ktd.z**4 ) * \
                np.exp(-(ktd.x**2 + ktd.y**2) / (2 * ktd.z**2 * self.mean_square_slope))   # Eq. 2.1.130   NB: k1^2 -> eps_2

        #return Tv, Th  #
        return coef*Tv, coef*Th


def shadow_function(mean_square_slope, cotan):

    # sqrt(1/pi) = 0.5641895835477563

    rel_cotan= cotan / (1.4142135623730951 * np.sqrt(mean_square_slope))  # sqrt(2)*s / mu, Eq. 2.1.154
    return 0.5 * (0.5641895835477563 / rel_cotan * np.exp(-rel_cotan**2) - scipy.special.erfc(rel_cotan))
