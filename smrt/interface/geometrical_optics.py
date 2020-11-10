

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
import scipy.integrate

from smrt.core.fresnel import fresnel_coefficients
from smrt.core.lib import smrt_matrix, abs2, generic_ft_even_matrix
from smrt.interface.vector3 import vector3
from smrt.core.interface import Interface


class GeometricalOptics(Interface):
    """A very rough surface.

"""
    args = ["mean_square_slope"]
    optional_args = {"shadow_correction": True}

    def clip_mu(self, mu):
        # avoid large zenith angles that causes many troubles
        return np.clip(mu, 0.1, 1)

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
        mu_i = np.atleast_1d(self.clip_mu(mu_i))[np.newaxis, np.newaxis, :]
        mu_s = np.atleast_1d(self.clip_mu(mu_s))[np.newaxis, :, np.newaxis]
        dphi = np.atleast_1d(dphi)[:, np.newaxis, np.newaxis]

        sin_i = np.sqrt(1 - mu_i**2)
        sin_s = np.sqrt(1 - mu_s**2)

        cos_phi = np.cos(dphi)
        sin_phi = np.sin(dphi)

        ki = vector3.from_xyz(sin_i, 0, -mu_i)
        ks = vector3.from_xyz(sin_s * cos_phi, sin_s * sin_phi, mu_s)

        # compute the local reflection Fresnel coefficient
        kd = ki - ks   # in principe: *sqrt(eps_1), but in the following it appears everywhere as a ratio

        n = kd / (np.sign(kd.z) * kd.norm)  # EQ 2.1.123 #equivalent to np.vector3(kd_x / kd_z, kd_y / kd_z, 1)

        mu_local = -vector3.dot(n, ki)
        assert(np.all(mu_local >= 0))
        assert(np.all(mu_local <= 1.0001))  # compare with some tolerance
        Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, self.clip_mu(mu_local))

        # define polarizations
        hs = vector3.from_xyz(-sin_phi, cos_phi, 0)
        vs = vector3.from_xyz(mu_s * cos_phi, mu_s * sin_phi, -sin_s)

        hi = vector3.from_xyz(0, 1, 0)
        vi = vector3.from_xyz(-mu_i, 0, -sin_i)

        # compute the dot products
        cross_ki_ks_norm = vector3.cross(ki, ks).norm
        colinear = cross_ki_ks_norm < 1e-4
        # avoid warning due to divide error, the colinear case is solved independantly:
        cross_ki_ks_norm[colinear] = 1

        hs_ki = vector3.dot(hs, ki) / cross_ki_ks_norm
        hs_ki[colinear] = -1
        vs_ki = vector3.dot(vs, ki) / cross_ki_ks_norm
        vs_ki[colinear] = 0

        hi_ks = vector3.dot(hi, ks) / cross_ki_ks_norm
        hi_ks[colinear] = 1

        vi_ks = vector3.dot(vi, ks) / cross_ki_ks_norm
        vi_ks[colinear] = 0

        fvv = abs2(hs_ki * hi_ks * Rh + vs_ki * vi_ks * Rv)   # Eqs 2.1.122 in Tsang_tomeIII
        fhh = abs2(vs_ki * vi_ks * Rh + hs_ki * hi_ks * Rv)

        fhv = abs2(vs_ki * hi_ks * Rh - hs_ki * vi_ks * Rv)
        fvh = abs2(hs_ki * vi_ks * Rh - vs_ki * hi_ks * Rv)

        reflection_coefficients = smrt_matrix.zeros((npol, npol, dphi.shape[0], mu_s.shape[1], mu_i.shape[2]))
        reflection_coefficients[0, 0] = fvv
        reflection_coefficients[0, 1] = fvh
        reflection_coefficients[1, 0] = fhv
        reflection_coefficients[1, 1] = fhh

        smrt_norm = 1 / (4 * np.pi)  # divide by 4*pi because this is the norm for SMRT

        coef = smrt_norm / (2 * self.mean_square_slope) / mu_i * kd.norm2**2 / kd.z**4 * \
            np.exp(-(kd.x**2 + kd.y**2) / (2 * kd.z**2 * self.mean_square_slope))  # Eq. 2.1.124


        if self.shadow_correction:
            backward = dphi == np.pi
            higher_thetas = mu_s <= mu_i
            zero_i = backward & higher_thetas
            zero_s = backward & ~higher_thetas
            # this hack to avoid division-by-zero is safe, because the shadow_function is only important for large angles
            sin_i[sin_i < 1e-3] = 1e-3
            sin_s[sin_s < 1e-3] = 1e-3

            s = 1 / (1 + (~zero_i) * shadow_function(self.mean_square_slope, mu_i / sin_i) +
                     (~zero_s) * shadow_function(self.mean_square_slope, mu_s / sin_s))
            coef *= s

        return reflection_coefficients * coef

    def ft_even_diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol):

        def reflection_function(dphi):
            return self.diffuse_reflection_matrix(frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol=npol)

        print("to be optimised")
        return generic_ft_even_matrix(reflection_function, m_max, nsamples=256)

    def ft_even_diffuse_transmission_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol):

        def transmission_function(dphi):
            return self.diffuse_transmission_matrix(frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol=npol)

        print("to be optimised")
        return generic_ft_even_matrix(transmission_function, m_max, nsamples=256)

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

        if abs(eta1_eta - 1) < 1e-6:
            raise NotImplementedError("the case of successive layers with identical index (%g) is not implemented" % n_2)
            return 1 / (4 * np.pi)   # return the identity matrix. The coef is to be checked.

        mu_i = np.atleast_1d(self.clip_mu(mu_i))[np.newaxis, np.newaxis, :]
        mu_t = np.atleast_1d(self.clip_mu(mu_t))[np.newaxis, :, np.newaxis]
        dphi = np.atleast_1d(dphi)[:, np.newaxis, np.newaxis]

        sin_i = np.sqrt(1 - mu_i**2)
        sin_t = np.sqrt(1 - mu_t**2)

        cos_phi = np.cos(dphi)
        sin_phi = np.sin(dphi)

        ki = vector3.from_xyz(sin_i, 0, -mu_i)
        kt = vector3.from_xyz(sin_t * cos_phi, sin_t * sin_phi, -mu_t)

        # compute the local transmission Fresnel coefficient
        ktd = ki * n_1.real - kt * n_2.real   # Eq 2.1.87

        n = ktd / (np.sign(ktd.z) * ktd.norm)  # Eq 2.1.128

        n_kt = -vector3.dot(n, kt)
        n_ki = -vector3.dot(n, ki)

        # compute Fresnel coefficients at stationary point
        Rh = (n_1.real * n_ki - n_2.real * n_kt) / (n_1.real * n_ki + n_2.real * n_kt)  # Eq. 2.1.132a
        Rv = (n_2.real * n_ki - n_1.real * n_kt) / (n_2.real * n_ki + n_1.real * n_kt)  # Eq. 2.1.132b

        no_compatible_slopes = (n_kt < 0) | (n_ki < 0)

        Rh[no_compatible_slopes] = -1
        Rv[no_compatible_slopes] = -1

        # define polarizations
        ht = vector3.from_xyz(-sin_phi, cos_phi, 0)
        vt = vector3.from_xyz(-mu_t * cos_phi, -mu_t * sin_phi, -sin_t)

        hi = vector3.from_xyz(0, 1, 0)
        vi = vector3.from_xyz(-mu_i, 0, -sin_i)

        # compute the cosines
        cross_ki_kt_norm = vector3.cross(ki, kt).norm
        colinear = cross_ki_kt_norm < 1e-4
        # avoid warning due to divide error, the colinear case is solved independantly:
        cross_ki_kt_norm[colinear] = 1

        ht_ki = vector3.dot(ht, ki) / cross_ki_kt_norm
        ht_ki[colinear] = -1  # checked with sympy

        vt_ki = vector3.dot(vt, ki) / cross_ki_kt_norm
        vt_ki[colinear] = 0  # checked with sympy

        hi_kt = vector3.dot(hi, kt) / cross_ki_kt_norm
        hi_kt[colinear] = 1    # checked with sympy
        vi_kt = vector3.dot(vi, kt) / cross_ki_kt_norm
        vi_kt[colinear] = 0    # checked with sympy

        Wvv = abs2(ht_ki * hi_kt * (1 + Rh) + vt_ki * vi_kt * (1 + Rv) * eta1_eta)   # Eqs 2.1.130 in Tsang_tomeIII
        Whh = abs2(vt_ki * vi_kt * (1 + Rh) + ht_ki * hi_kt * (1 + Rv) * eta1_eta)

        Whv = abs2(-vt_ki * hi_kt * (1 + Rh) + ht_ki * vi_kt * (1 + Rv) * eta1_eta)
        Wvh = abs2(ht_ki * vi_kt * (1 + Rh) - vt_ki * hi_kt * (1 + Rv) * eta1_eta)

        transmission_coefficients = smrt_matrix.zeros((npol, npol, dphi.shape[0], mu_t.shape[1], mu_i.shape[2]))

        transmission_coefficients[0, 0] = Wvv
        transmission_coefficients[0, 1] = Wvh
        transmission_coefficients[1, 0] = Whv
        transmission_coefficients[1, 1] = Whh

        smrt_norm = 1 / (4 * np.pi)   # SMRT requires scattering coefficient / 4 * pi

        coef = smrt_norm * 2 * eps_2 * ktd.norm2 * n_kt**2 / (eta1_eta * self.mean_square_slope * mu_i * ktd.z**4) * \
            np.exp(-(ktd.x**2 + ktd.y**2) / (2 * ktd.z**2 * self.mean_square_slope))   # Eq. 2.1.130   NB: k1^2 -> eps_2

        if self.shadow_correction:
            # this hack to avoid division-by-zero is safe, because the shadow_function is only important for large angles
            sin_i[sin_i < 1e-3] = 1e-3
            sin_t[sin_t < 1e-3] = 1e-3
            s = 1 / (1 + shadow_function(self.mean_square_slope, mu_i / sin_i) + shadow_function(self.mean_square_slope, mu_t / sin_t))
            coef *= s

        return transmission_coefficients * coef.real

    def reflection_integrand_for_energy_conservation_test(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi):
        """function relevant to compute energy conservation. See p87 in Tsang_tomeIII.
"""
        mu_i = np.atleast_1d(self.clip_mu(mu_i))[np.newaxis, np.newaxis, :]
        mu_s = np.atleast_1d(self.clip_mu(mu_s))[np.newaxis, :, np.newaxis]
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
        hi = vector3.from_xyz(0, 1, 0)
        vi = vector3.from_xyz(-mu_i, 0, -sin_i)

        hi_ks = vector3.dot(hi, ks)
        vi_ks = vector3.dot(vi, ks)

        coef = 1 / (2 * np.pi * self.mean_square_slope) * kd.norm2**2 / (4 * mu_i * vector3.cross(ki, ks).norm2 * kd.z**4) * \
            np.exp(- (kd.x**2 + kd.y**2) / (2 * kd.z**2 * self.mean_square_slope))  # Eq. 2.1.124

        return coef * (hi_ks**2 * abs2(Rh) + vi_ks**2 * abs2(Rv)), \
            coef * (vi_ks**2 * abs2(Rh) + hi_ks**2 * abs2(Rv))

    def transmission_integrand_for_energy_conservation_test(self, frequency, eps_1, eps_2, mu_t, mu_i, dphi):
        """function relevant to compute energy conservation. See p87 in Tsang_tomeIII.
"""
        n_2 = np.sqrt(eps_2)
        n_1 = np.sqrt(eps_1)

        mu_i = np.atleast_1d(self.clip_mu(mu_i))[np.newaxis, np.newaxis, :]
        mu_t = np.atleast_1d(self.clip_mu(mu_t))[np.newaxis, :, np.newaxis]
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

        # compute Fresnel coefficients at stationary point
        n_kt = -vector3.dot(n, kt)
        n_ki = -vector3.dot(n, ki)

        Rh = (n_1.real * n_ki - n_2.real * n_kt) / (n_1.real * n_ki + n_2.real * n_kt)  # Eq. 2.1.132a
        Rv = (n_2.real * n_ki - n_1.real * n_kt) / (n_2.real * n_ki + n_1.real * n_kt)  # Eq. 2.1.132b

        no_compatible_slopes = (n_kt < 0) | (n_ki < 0)

        Rh[no_compatible_slopes] = -1
        Rv[no_compatible_slopes] = -1

        # define polarizations

        hi = vector3.from_xyz(0, 1, 0)
        vi = vector3.from_xyz(-mu_i, 0, -sin_i)

        hi_kt = vector3.dot(hi, kt)
        vi_kt = vector3.dot(vi, kt)

        Tv = (hi_kt**2) * (1 - abs2(Rh)) + (vi_kt**2) * (1 - abs2(Rv))
        Th = (vi_kt**2) * (1 - abs2(Rh)) + (hi_kt**2) * (1 - abs2(Rv))

        coef = eps_2 / (2 * np.pi * self.mean_square_slope) * ktd.norm2 * n_kt * n_ki  \
            / (mu_i * vector3.cross(ki, kt).norm2 * ktd.z**4) * \
            np.exp(- (ktd.x**2 + ktd.y**2) / (2 * ktd.z**2 * self.mean_square_slope))   # Eq. 2.1.130 NB: k1^2 -> eps_2

        if self.shadow_correction:
            sin_i[sin_i < 1e-3] = 1e-3
            sin_t[sin_t < 1e-3] = 1e-3
            s = 1 / (1 + shadow_function(self.mean_square_slope, mu_i / sin_i) + shadow_function(self.mean_square_slope, mu_t / sin_t))
            coef *= s

        return coef * Tv, coef * Th

    def reflection_coefficients(self, frequency, eps_1, eps_2, mu_i):
        # for debugging only at this stage
        n_mu = 512 + 1
        n_phi = 128

        mu = np.linspace(1e-7, 1, n_mu, endpoint=True)
        dphi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

        R = self.diffuse_reflection_matrix(10e9, eps_1, eps_2, mu, mu_i, dphi, 2)

        return self._integrate_coefficients(mu, dphi, R)

    def transmission_coefficients(self, frequency, eps_1, eps_2, mu_i):
        # for debugging only at this stage
        n_mu = 512 + 1
        n_phi = 128

        mu = np.linspace(1e-7, 1, n_mu, endpoint=True)
        dphi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

        T = self.diffuse_transmission_matrix(10e9, eps_1, eps_2, mu, mu_i, dphi, 2)

        return self._integrate_coefficients(mu, dphi, T)

    def _integrate_coefficients(self, mu, dphi, x):

        # integrate the pola first, then the azimuth and last the mu
        x = x.values.sum(axis=(0, 2))

        # x is not pola_inc, mu_inc
        # return scipy.integrate.simps(x, mu, axis=0) * (dphi[1] - dphi[0])  # use simpson method if n_mu is not 2**n + 1
        return scipy.integrate.romb(x, dx=mu[1] - mu[0], axis=1) * (dphi[1] - dphi[0])


def shadow_function(mean_square_slope, cotan):

    # sqrt(1/pi) = 0.5641895835477563

    rel_cotan = cotan / (1.4142135623730951 * np.sqrt(mean_square_slope))  # sqrt(2)*s / mu, Eq. 2.1.154
    return 0.5 * (0.5641895835477563 / rel_cotan * np.exp(-rel_cotan**2) - scipy.special.erfc(rel_cotan))
