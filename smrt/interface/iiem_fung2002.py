"""
Provide the interface boundary condition under IIEM (Improved IEM) formulation provided by Fung et al. 2002.

The extended domain of validity (for large roughness or correlation length) is produced by using the transition Fresnel
coefficients (Fung et al. 2004). This code also produces bi-static coefficients for passive sensor and second order interaction with
snow volume. Multiple scattering for crosspol is implemented from the original formulation in Fung 92. The integral for multiple
scattering is done by fixed order quadrature for faster computation. A more complex implementation would be AIEM (Wu et al 2004).

Notes:
    Compute diffuse reflection as described in Fung et al. 2002, the specular reflection
    and the coherent transmission.  The implementation is only valid for the substrate has
    it does not provide the diffuse transmission.

Usage:
    Basic usage with default settings:
        >>> soil = make_soil('iiem_fung2002', complex(2, 0.01), roughness_rms = 0.001, corr_length = 0.02, temperature = 265)

Credit:
    This code was based on the MATLAB code published by Ulaby & Long, 2014: https://tools.grss-ieee.org/rscl1/coderecord.php?id=469
    and Robbie Mallet's python version: https://github.com/robbiemallett/IIEM

References:
    Fung. A.K., Liu, W.Y., Chen, K.S., & Tsay, M.K. (2002). An Improved Iem Model for Bistatic Scattering From Rough Surfaces. Journal
    of Electromagnetic Waves and Applications. 16(5), 689-702. https://doi.org/10.1163/156939302X01119

    Fung. A. K., & Chen, K. S. (2004). An Update on the IEM Surface Backscattering Model. IEEE. 1(2), 75-77.
    https://doi.org/10.1109/LGRS.2004.826564
"""

import numpy as np

# local import
from smrt.core.error import SMRTError
from smrt.core.fresnel import fresnel_coefficients
from smrt.core.globalconstants import C_SPEED
from smrt.core.lib import abs2, cached_roots_legendre, generic_ft_even_matrix, smrt_matrix
from smrt.core.vector3 import vector3
from smrt.interface.geometrical_optics import _clip_mu, shadow_function
from smrt.interface.iem_fung92 import IEM_Fung92


class IIEM_Fung2002(IEM_Fung92):
    """
    Implement a moderate rough surface model for passive and active. Provide bi-static coefficient

    Multiple scattering only for crosspol backscatter since it's assumed to be negligeable for co pol (passive??? to be implemented).
    Use with care


    Args:
        roughness_rms: Root-Mean-Squared surface roughness.
        corr_length: Correlation length of the surface.
        autocorrelation_function: [Optional] Type of autocorrelation function to use. Default is "exponential".
        warning_handling: [Optional] Parameter that dictates how to handle wanring. Default is "print".
        series_truncation: [Optional] Number of iterations to use in the summation of roughness spectra.
        N_integral: [Optional] Number of streams to use in the integral for multiple scattering.
        shadow_correction: [Optional] Use a shadow correction of the rough surface when dealing with significant surface roughness or
            large scattering angles. Default is ``True``.
        compute_crosspol: [Optional] Compute the multiple scattering for cross-pol. Default is ``True``.
        transition_fresnel: [Optional] Use transitionnal Fresnel coefficients define in Fung et al. (2004) Default is ``True``.
    """

    optional_args = {
        "autocorrelation_function": "exponential",
        "warning_handling": "print",
        "series_truncation": 10,
        "N_integral": 20,  # number of fixed quadrature for integral
        "shadow_correction": True,
        "compute_crosspol": True,  # set False to disable cross-pol calculation
        "transition_fresnel": True,
    }

    def check_validity(self, ks):
        # check validity
        if ks > 3:
            raise SMRTError(
                "Warning, roughness_rms is too high for the given wavelength. Limit is ks < 3. Here ks=%g" % ks
            )

    def transition_fresnel_coefficients(self, eps_1, eps_2, mu_i, k, k_w, n):
        """
        Calculate the transition Fresnel coefficients for IIEM (see Fung et al 2004)
        """
        eps_r = eps_2.real

        # at 0
        Rv_0, Rh_0, _ = fresnel_coefficients(eps_1, eps_2, 1)
        # at mu
        Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_i)

        # sin_ i squared
        sin_i2 = 1 - mu_i**2
        Fv = 8 * abs2(Rv_0) * sin_i2 * ((mu_i + np.sqrt(eps_r - sin_i2)) / (mu_i * np.sqrt(eps_r - sin_i2)))
        Fh = 8 * abs2(Rh_0) * sin_i2 * ((mu_i + np.sqrt(eps_r - sin_i2)) / (mu_i * np.sqrt(eps_r - sin_i2)))

        Sv_0 = 1 / abs2(1 + (8 * Rv_0) / (Fv * mu_i))
        Sh_0 = 1 / abs2(1 + (8 * Rh_0) / (Fh * mu_i))

        rms_mu_over_factorial = np.cumprod((k.norm() * self.roughness_rms * mu_i) ** (2) / n, axis=-1)

        factor_Rv0 = 2 ** (n + 1) * Rv_0 * np.exp(-((k.norm() * self.roughness_rms * mu_i) ** 2)) / mu_i
        factor_Rh0 = 2 ** (n + 1) * Rh_0 * np.exp(-((k.norm() * self.roughness_rms * mu_i) ** 2)) / mu_i

        Sv_n = np.sum(abs2(Fv) / 4 * rms_mu_over_factorial * self.W_n(n, k_w), axis=-1, keepdims=True)
        Sv_d = np.sum((rms_mu_over_factorial * abs2(Fv / 2 + factor_Rv0) * self.W_n(n, k_w)), axis=-1, keepdims=True)

        Sh_n = np.sum(abs2(Fh) / 4 * rms_mu_over_factorial * self.W_n(n, k_w), axis=-1, keepdims=True)
        Sh_d = np.sum((rms_mu_over_factorial * abs2(Fh / 2 + factor_Rh0) * self.W_n(n, k_w)), axis=-1, keepdims=True)

        Sv = Sv_n / Sv_d
        Sh = Sh_n / Sh_d

        gamma_v = 1 - (Sv / Sv_0)
        gamma_h = 1 - (Sh / Sh_0)

        Rv_t = Rv + (Rv_0 - Rv) * gamma_v
        Rh_t = Rh + (Rh_0 - Rh) * gamma_h

        return Rv_t, Rh_t

    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        """
        Compute the diffuse reflection coefficients.

        Coefficients are calculated for an array of incident, scattered and azimuth angles in medium 1. Medium 2 is where the
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

        mu_i = np.atleast_1d(_clip_mu(mu_i))[np.newaxis, np.newaxis, :, np.newaxis]
        mu_s = np.atleast_1d(_clip_mu(mu_s))[np.newaxis, :, np.newaxis, np.newaxis]
        dphi = np.atleast_1d(dphi)[:, np.newaxis, np.newaxis, np.newaxis]

        # incident wavenumber
        k = vector3.from_angles(2 * np.pi * frequency / C_SPEED * np.sqrt(eps_1).real, mu_i, 0)
        # scattered wavenumber
        k_s = vector3.from_angles(2 * np.pi * frequency / C_SPEED * np.sqrt(eps_1).real, mu_s, dphi)

        # wavenumber for roughness spectra
        # k_w is 1d representation of W_n(ksx - kx, ksy - ky) eqn 4 Fung et al 2002
        # k * np.sqrt((sin_s * cos_phi_s - sin_i * cos_phi_i) ** 2 + (sin_s * sin_phi_s - sin_i * sin_phi_i) ** 2)) # from ulaby code
        # phi_i = 0
        # phi_s = dphi = phi_s - 0
        # k_w with dphi assumimg phi_i = 0
        # (sin_s * cos_dphi - sin_i) ** 2 + (sin_s * sin_dphi) ** 2)

        sin_i = np.sqrt(1 - mu_i**2)
        sin_s = np.sqrt(1 - mu_s**2)
        cos_dphi = np.cos(dphi)
        sin_dphi = np.sqrt(1 - cos_dphi**2)
        k_w = k.norm() * np.sqrt((sin_s * cos_dphi - sin_i) ** 2 + (sin_s * sin_dphi) ** 2)

        ks = np.abs(k.norm() * self.roughness_rms)
        # kl = np.abs(k.norm() * self.corr_length)

        try:
            self.check_validity(ks)
        except SMRTError as e:
            if self.warning_handling == "print":
                print(e)
            elif self.warning_handling == "nan":
                return smrt_matrix.full((npol, len(mu_i)), np.nan)

        # prepare the series
        N = self.series_truncation
        n = np.arange(1, N + 1, dtype=np.float64)  # [:, None]
        n = np.atleast_1d(n)[np.newaxis, np.newaxis, np.newaxis, :]

        rms2 = self.roughness_rms**2
        rms2_over_fractorial = np.cumprod(rms2 / n, axis=-1)  # [:, None]

        # transition fresnel
        if self.transition_fresnel:
            Rv, Rh = self.transition_fresnel_coefficients(eps_1, eps_2, mu_i, k, k_w, n)
        else:
            Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_i)

        # #Debug, R from matlab code
        # rt = np.sqrt(eps_2 - sin_i**2)
        # Rv = (eps_2 * mu_i - rt) / (eps_2 * mu_s + rt)
        # Rh = (mu_i - rt) / (mu_i + rt)

        Ivv_n, Ihh_n = calculate_Iqp(eps_1, eps_2, k.norm(), k.z, k_s.z, Rv, Rh, n, mu_i, mu_s, dphi, rms2)

        # Eq 14 in wu et al. 2004
        coef = k.norm2() / 2 * np.exp(-rms2 * (k.z**2 + k_s.z**2))
        coef_n = rms2_over_fractorial * self.W_n(n, k_w)

        if self.shadow_correction:
            sin_i[sin_i < 1e-3] = 1e-3
            sin_s[sin_s < 1e-3] = 1e-3
            mean_square_slope = (self.roughness_rms / self.corr_length) ** 2
            # equation 7 from fung et al 2002
            s = 1 / (
                1 + shadow_function(mean_square_slope, mu_i / sin_i) + shadow_function(mean_square_slope, mu_s / sin_s)
            )
            coef *= s

        reflection_coefficients = smrt_matrix.zeros((npol, npol, dphi.shape[0], mu_s.shape[1], mu_i.shape[2]))

        reflection_coefficients[0, 0] = np.sum(coef * coef_n * abs2(Ivv_n) / (4 * np.pi * mu_i), axis=-1)
        reflection_coefficients[1, 1] = np.sum(coef * coef_n * abs2(Ihh_n) / (4 * np.pi * mu_i), axis=-1)

        # only for backscatter
        # calculate multiple scattering contribution for cross-pol backscatter with a double integral function
        if self.compute_crosspol:
            # take regular fresnel, not transitionnal... not valid for cross for now
            Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_i)
            Rvh = (Rv - Rh) / 2
            ks2 = ks**2
            svh = self.double_integral(k, ks2, mu_i, eps_2, Rvh, n, self.N_integral)
            # reshape to match angles shape set initially
            svh = svh.reshape(1, 1, mu_i.shape[2], 1)

            if self.shadow_correction:
                s = 1 / (1 + shadow_function(mean_square_slope, mu_i / sin_i) * 2)
                svh *= s

            # reshape again to match result final shape
            svh = svh.reshape(1, 1, mu_i.shape[2])
            reflection_coefficients[0, 1] = svh / (4 * np.pi * mu_i)
            reflection_coefficients[1, 0] = svh / (4 * np.pi * mu_i)

        return reflection_coefficients

    def ft_even_diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol):
        def reflection_function(dphi):
            return self.diffuse_reflection_matrix(frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol=npol)

        return generic_ft_even_matrix(reflection_function, m_max, nsamples=256)

    def diffuse_transmission_matrix(self, frequency, eps_1, eps_2, mu_t, mu_i, dphi, npol):
        """
        Compute the transmission coefficients for the azimuthal mode m and for an array of incidence angles (given by their cosine)
        in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium
        :param mu1: array of cosine of incident angles
        :param npol: number of polarization

        :return: the transmission matrix
        """

        return NotImplementedError(
            "The use of the iiem is restricted to substrate only for now,"
            " Missing the implementation of the diffuse transmission"
        )

    def W_n_2D(self, n, k, rx, ry, sin_i):
        """
        Calculate the 2-D roughness spectra for n (multiple scattering)
        """

        kl2 = (k.norm() * self.corr_length) ** 2

        if self.autocorrelation_function == "gaussian":
            w_n = 0.5 * kl2 / n * np.exp(-kl2 * ((rx - sin_i) ** 2 + ry**2) / (4 * n))
            return w_n

        elif self.autocorrelation_function == "exponential":
            w_n = n * kl2 / (n**2 + kl2 * ((rx - sin_i) ** 2 + ry**2)) ** 1.5
            return w_n

        else:
            raise SMRTError("The autocorrelation function must be expoential or gaussian")

    def W_m_2D(self, n, k, rx, ry, sin_i):
        """
        Calculate the 2-D roughness spectra for m (multiple scattering)
        """
        kl2 = (k.norm() * self.corr_length) ** 2

        if self.autocorrelation_function == "gaussian":
            w_n = 0.5 * kl2 / n * np.exp(-kl2 * ((rx + sin_i) ** 2 + ry**2) / (4 * n))
            return w_n

        elif self.autocorrelation_function == "exponential":
            w_n = n * kl2 / (n**2 + kl2 * ((rx + sin_i) ** 2 + ry**2)) ** 1.5
            return w_n

        else:
            raise SMRTError("The autocorrelation function must be exponential or gaussian")

    def xpol_integralfunction(self, r, dphi, k, ks2, mu_i, eps_2, Rvh, n):
        # expand dim to accomodate all variables
        # summation count
        m = n.reshape(1, 1, n.shape[-1], 1, 1)
        n = n.reshape(1, n.shape[-1], 1, 1, 1)
        # multiple angles
        mu_i = mu_i.reshape(mu_i.shape[2], 1, 1, 1, 1)
        Rvh = Rvh.reshape(Rvh.shape[2], 1, 1, 1, 1)
        # integral variables

        r = r.reshape(1, 1, 1, r.shape[0], r.shape[1])
        dphi = dphi.reshape(1, 1, 1, dphi.shape[0], dphi.shape[1])

        mu_i2 = mu_i**2
        sin_i = np.sqrt(1 - mu_i2)
        cos_dphi = np.cos(dphi)
        sin_dphi = np.sqrt(1 - cos_dphi**2)
        rx = r * cos_dphi
        ry = r * sin_dphi
        r2 = r**2

        # calculation of the field coefficients
        q = np.sqrt(1.0001 - r2)
        qt = np.sqrt(eps_2 - r2)

        a = (1 + Rvh) / q
        b = (1 - Rvh) / q
        c = (1 + Rvh) / qt
        d = (1 - Rvh) / qt

        # calculate cross-pol coefficient
        # reorganised from eqn A28 Fung et al 1992
        B3 = rx * ry / mu_i
        fvh1 = (b - c) * (1 - 3 * Rvh) - (b - c / eps_2) * (1 + Rvh)
        fvh2 = (a - d) * (1 + 3 * Rvh) - (a - d * eps_2) * (1 - Rvh)
        Fvh = abs2((fvh1 + fvh2) * B3)

        # # # calculate shadowing func for multiple scattering
        rms_slope = self.roughness_rms / self.corr_length
        sha = 1 / (1 + shadow_function(rms_slope**2, q / r))

        # calculate expressions for the surface spectra
        w_n = self.W_n_2D(n, k, rx, ry, sin_i)
        w_m = self.W_m_2D(m, k, rx, ry, sin_i)

        # --compute VH scattering coefficient
        vh_coef = np.exp(-2 * ks2 * mu_i2) / (16 * np.pi)
        vhmnsum = w_n * w_m * (ks2 * mu_i2) ** (n + m) / np.cumprod(n, axis=1) / np.cumprod(m, axis=2)
        # sum over axis for n and  m
        VH = np.sum(4 * vh_coef * Fvh * vhmnsum * r * sha, axis=(1, 2))
        return VH

    def double_integral(self, k, ks2, mu_i, eps_2, Rvh, n, n_order):
        """
        Double integral function that is vectorized to handle multidimensionnal integrand (mu_i)
        Using Gauss legendre polynomials to do a fixed order quadrature
        """
        # can handle multidimensionnal integrand
        # Integration bounds
        a_r, b_r = 0.1, 1.0
        a_phi, b_phi = 0.0, np.pi

        # Get Gauss-Legendre nodes and weights
        nodes_r, weights_r = cached_roots_legendre(n_order)
        nodes_phi, weights_phi = cached_roots_legendre(n_order)

        # Rescale from [-1, 1] to [a, b]
        r = 0.5 * (nodes_r + 1) * (b_r - a_r) + a_r
        phi = 0.5 * (nodes_phi + 1) * (b_phi - a_phi) + a_phi
        wr = 0.5 * (b_r - a_r) * weights_r
        wphi = 0.5 * (b_phi - a_phi) * weights_phi

        # Create 2D meshgrid of r and phi shape: (n_order, n_order)
        R, PHI = np.meshgrid(r, phi, indexing="ij")
        WR, WPHI = np.meshgrid(wr, wphi, indexing="ij")

        # Evaluate integrand on the whole grid # shape (mu_i, n_order, n_order)
        integrand_vals = self.xpol_integralfunction(R, PHI, k=k, ks2=ks2, mu_i=mu_i, eps_2=eps_2, Rvh=Rvh, n=n)

        # Multiply integrand by weights and sum over phi and r
        integral_result = np.sum(integrand_vals * WR * WPHI, axis=(1, 2))

        return integral_result


def calculate_F(ud, is_, Rv, Rh, eps_r, k_norm, kz, k_sz, mu_i, mu_s, dphi):
    """
    Calculate propagating field coefficients (F) and c_i coefficients, eqn 2 and 3 in Fung et al 2002.
    Code modified from ulaby et al 2014 matlab code and Robbie Mallet https://github.com/robbiemallett/IIEM
    """
    # geometry
    sin_i = np.sqrt(1 - mu_i**2)
    sin_s = np.sqrt(1 - mu_s**2)
    cos_phi_i = 1.0  # np.cos(0)
    # sin_phi_i = 0 #np.sin(0)
    cos_dphi = np.cos(dphi)
    sin_dphi = np.sqrt(1 - cos_dphi**2)

    # clip to avoid negative sqrt
    eps_r_sin_i2 = np.clip(eps_r - sin_i**2, 0.01, eps_r)
    sin_cosdphi_diff = sin_s * cos_dphi - sin_i * cos_phi_i
    knorm_sin_allangle_squared = k_norm * sin_i * sin_s * sin_dphi**2

    if is_ == 1:
        Gqi = ud * kz
        Gqti = ud * k_norm * np.sqrt(eps_r_sin_i2)
        qi = ud * kz

        c11 = k_norm * cos_dphi * (k_sz - qi)
        c21 = mu_i * (
            cos_dphi * (k_norm**2 * sin_i * cos_phi_i * (sin_cosdphi_diff) + Gqi * (k_norm * mu_s - qi))
            + k_norm**2 * cos_phi_i * sin_i * sin_s * sin_dphi**2
        )
        c31 = (
            k_norm
            * sin_i
            * (
                sin_i * cos_phi_i * cos_dphi * (k_norm * mu_s - qi)
                - Gqi * (cos_dphi * (sin_cosdphi_diff) + sin_s * sin_dphi**2)
            )
        )
        c41 = k_norm * mu_i * (cos_dphi * mu_s * (k_norm * mu_s - qi) + k_norm * sin_s * (sin_cosdphi_diff))
        c51 = Gqi * (cos_dphi * mu_s * (qi - k_norm * mu_s) - k_norm * sin_s * (sin_cosdphi_diff))

        c12 = k_norm * cos_dphi * (k_sz - qi)
        c22 = mu_i * (
            cos_dphi * (k_norm**2 * sin_i * cos_phi_i * (sin_cosdphi_diff) + Gqti * (k_norm * mu_s - qi))
            + k_norm**2 * cos_phi_i * sin_i * sin_s * sin_dphi**2
        )
        c32 = (
            k_norm
            * sin_i
            * (
                sin_i * cos_phi_i * cos_dphi * (k_norm * mu_s - qi)
                - Gqti * (cos_dphi * (sin_cosdphi_diff) - sin_s * sin_dphi**2)
            )
        )
        # c42 = c41
        c52 = Gqti * (cos_dphi * mu_s * (qi - k_norm * mu_s) - k_norm * sin_s * (sin_cosdphi_diff))

    if is_ == 2:
        Gqs = ud * k_sz
        Gqts = ud * k_norm * np.sqrt(eps_r_sin_i2)
        qs = ud * k_sz

        c11 = k_norm * cos_dphi * (kz + qs)
        c21 = Gqs * (
            cos_dphi * (mu_i * (k_norm * mu_i + qs) - k_norm * sin_i * (sin_cosdphi_diff)) - knorm_sin_allangle_squared
        )
        c31 = k_norm * sin_s * (k_norm * mu_i * (sin_cosdphi_diff) + sin_i * (kz + qs))
        c41 = (
            k_norm
            * mu_s
            * (cos_dphi * (mu_i * (kz + qs) - k_norm * sin_i * (sin_cosdphi_diff)) - knorm_sin_allangle_squared)
        )
        c51 = -mu_s * (k_norm**2 * sin_s * (sin_cosdphi_diff) + Gqs * cos_dphi * (kz + qs))

        c12 = k_norm * cos_dphi * (kz + qs)
        c22 = Gqts * (cos_dphi * (mu_i * (kz + qs) - k_norm * sin_i * (sin_cosdphi_diff)) - knorm_sin_allangle_squared)
        c32 = k_norm * sin_s * (k_norm * mu_i * (sin_cosdphi_diff) + sin_i * (kz + qs))
        # c42 = c41
        c52 = -mu_s * (k_norm**2 * sin_s * (sin_cosdphi_diff) + Gqts * cos_dphi * (kz + qs))

    q = kz
    qt = k_norm * np.sqrt(eps_r_sin_i2)

    Fvv = (
        (1 + Rv) * (-(1 - Rv) * c11 / q + (1 + Rv) * c12 / qt)
        + (1 - Rv) * ((1 - Rv) * c21 / q - (1 + Rv) * c22 / qt)
        + (1 + Rv) * ((1 - Rv) * c31 / q - (1 + Rv) * c32 / eps_r / qt)
        + (1 - Rv) * ((1 + Rv) * c41 / q - eps_r * (1 - Rv) * c41 / qt)
        + (1 + Rv) * ((1 + Rv) * c51 / q - (1 - Rv) * c52 / qt)
    )

    Fhh = (
        (1 + Rh) * ((1 - Rh) * c11 / q - eps_r * (1 + Rh) * c12 / qt)
        - (1 - Rh) * ((1 - Rh) * c21 / q - (1 + Rh) * c22 / qt)
        - (1 + Rh) * ((1 - Rh) * c31 / q - (1 + Rh) * c32 / qt)
        - (1 - Rh) * ((1 + Rh) * c41 / q - (1 - Rh) * c41 / qt)
        - (1 + Rh) * ((1 + Rh) * c51 / q - (1 - Rh) * c52 / qt)
    )

    return Fvv, Fhh


def calculate_Iqp(eps_1, eps_2, k_norm, kz, k_sz, Rv, Rh, n, mu_i, mu_s, dphi, rms2):
    """
    Calculate Iqp, eqn 5 in Fung et al 2002.
    """
    eps_r = eps_2.real / eps_1.real

    sin_i = np.sqrt(1 - mu_i**2)
    sin_s = np.sqrt(1 - mu_s**2)

    fvv = 2 * Rv / (mu_i + mu_s) * (sin_i * sin_s - (1 + mu_i * mu_s) * np.cos(dphi))  # Eq 5 in Fung et al. 2002
    fhh = -2 * Rh / (mu_i + mu_s) * (sin_i * sin_s - (1 + mu_i * mu_s) * np.cos(dphi))  # Eq 5 in Fung et al. 2002
    # fvh, fhv = 0

    # Calculate the Field coefficients
    Fvv_up_i, Fhh_up_i = calculate_F(+1, 1, Rv, Rh, eps_r, k_norm, kz, k_sz, mu_i, mu_s, dphi)
    Fvv_up_s, Fhh_up_s = calculate_F(+1, 2, Rv, Rh, eps_r, k_norm, kz, k_sz, mu_i, mu_s, dphi)
    Fvv_dn_i, Fhh_dn_i = calculate_F(-1, 1, Rv, Rh, eps_r, k_norm, kz, k_sz, mu_i, mu_s, dphi)
    Fvv_dn_s, Fhh_dn_s = calculate_F(-1, 2, Rv, Rh, eps_r, k_norm, kz, k_sz, mu_i, mu_s, dphi)

    kirch_vv = (k_sz + kz) ** (n) * fvv * np.exp(-rms2 * kz * k_sz)
    kirch_hh = (k_sz + kz) ** (n) * fhh * np.exp(-rms2 * kz * k_sz)

    A_vv = (
        (k_sz - kz) ** (n - 1) * Fvv_up_i * np.exp(-rms2 * (kz**2 - kz * (k_sz - kz)))
        + (k_sz + kz) ** (n - 1) * Fvv_dn_i * np.exp(-rms2 * (kz**2 + kz * (k_sz - kz)))
        + (kz + k_sz) ** (n - 1) * Fvv_up_s * np.exp(-rms2 * (k_sz**2 - k_sz * (k_sz - kz)))
        + (kz - k_sz) ** (n - 1) * Fvv_dn_s * np.exp(-rms2 * (k_sz**2 + k_sz * (k_sz - kz)))
    )

    A_hh = (
        (k_sz - kz) ** (n - 1) * Fhh_up_i * np.exp(-rms2 * (kz**2 - kz * (k_sz - kz)))
        + (k_sz + kz) ** (n - 1) * Fhh_dn_i * np.exp(-rms2 * (kz**2 + kz * (k_sz - kz)))
        + (kz + k_sz) ** (n - 1) * Fhh_up_s * np.exp(-rms2 * (k_sz**2 - k_sz * (k_sz - kz)))
        + (kz - k_sz) ** (n - 1) * Fhh_dn_s * np.exp(-rms2 * (k_sz**2 + k_sz * (k_sz - kz)))
    )

    Ivv_n = kirch_vv + A_vv / 4
    Ihh_n = kirch_hh + A_hh / 4

    return Ivv_n, Ihh_n
