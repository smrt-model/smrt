"""Provide the interface boundary condition under the Small Perturbation Method (SPM) formulation as described by
Iodice, A., & Imperatore, P. 2024

Notes:
    This implementation computes the bistatic diffuse reflection based on the SPM formulation in Iodice, A., &
    Imperatore, P. 2024. The SPM is an analytical solution valid under the assumption of very small surface roughness,
    and is therefore only applicable when the surface height perturbations are small compared to the wavelength,
    typically for ks < 0.3, where k is the wavenumber and s is the rms height of the surface. sqrt(2)*s/l < 0.3 is also
    recommended, where l is the correlation length of the surface.

    As a result, this approach is suitable primarily for backscatter calculations or scenarios where the surface is very
    smooth or low frequencies (longer wavelengths).

    In principle in SPM the zeroth-order is the fresnel reflection without attenuation of the Kirchoff approximation.
    However,  we keep it here even though this attenuation should be negligible because by requirement of the
    approximation ks << 1.

**Usage**::

    # rms height and corr_length values work at 10 GHz
    substrate = make_soil_substrate("small_pertubation_method", "dobson85_peplinski95", temperature=260,
                                            roughness_rms=1e-3,
                                            corr_length=5e-2,
                                            autocorrelation_function="exponential",
                                            moisture=moisture,
                                            clay=clay, sand=sand, drymatter=drymatter)

References:
    Iodice, A., & Imperatore, P. 2024. A Simple Path to the Small Perturbation Method for Scattering from Slightly Rough
    Dielectric Surfaces. Remote Sensing, 16(16), 3035. https://doi.org/10.3390/rs16163035

    Koudogbo, F., Combes, P. F., & Mametsa, H.-J. (2004). numerical and experimental validations of iem for bistatic
    scattering from natural and manmade rough surfaces. Progress in Electromagnetics Research, 46, 203–244.
    https://doi.org/10.2528/pier03092902
"""

import numpy as np

from smrt.core.error import SMRTError
from smrt.core.fresnel import (
    fresnel_reflection_coefficients,
)
from smrt.core.globalconstants import TWO_PI, WAVENUMBER_PER_HZ
from smrt.core.interface import Interface
from smrt.core.lib import abs2, generic_ft_even_matrix, smrt_matrix, vectorize_angles
from smrt.interface.interface_utils import HemisphericalIntegrationMixin, KirchoffApproximationCoherentInterfaceMixin


class SmallPerturbationMethod(KirchoffApproximationCoherentInterfaceMixin, HemisphericalIntegrationMixin, Interface):
    """Implement the Small Perturbation Method.

    [[It is not suitable for emissivity calculations. Use with care!]]

    Args:
        roughness_rms: Root-Mean-Squared surface roughness.
        corr_length: Correlation length of the surface.
        autocorrelation_function: [Optional] Type of autocorrelation function to use. Default is "exponential".
        warning_handling: [Optional] Parameter that dictates how to handle wanring. Default is "print".
        series_truncation: [Optional] Number of iterations to use in the summation of roughness spectra.
    """

    args = ["roughness_rms", "corr_length"]
    optional_args = {
        "autocorrelation_function": "exponential",
    }

    def check_validity(self, ks, kl, eps_r):
        # check validity
        if ks > 1:
            raise SMRTError(
                "Warning, roughness_rms is too high for the given wavelength. "
                f"Recommended is ks < 0.3 and the limit is ks < 1. Here ks={ks}."
            )
        if np.sqrt(2) * ks / kl < 0.3:
            raise SMRTError(
                "Warning, slope (roughness_rms / correlation_length) is too high. "
                f"Recommended is  ks / kl  < 0.3 / sqrt(2) . Here ks/kl={ks / kl}."
            )

    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol, debug=False):
        """Compute the diffuse reflection coefficients.

        Coefficients are calculated for an array of incident, scattered and azimuth angles in medium 1. Medium 2 is
        where the beam is transmitted.

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
        mu_s, sin_s, mu_i, sin_i, cosphi, sinphi = vectorize_angles(mu_s, mu_i, dphi)

        rvs, rhs, _ = fresnel_reflection_coefficients(eps_1, eps_2, mu_s)
        rvi, rhi, _ = fresnel_reflection_coefficients(eps_1, eps_2, mu_i)

        eps_r = eps_2 / eps_1
        deps = eps_r - 1

        # alpha
        ahh = deps * (1 + rhi) * (1 + rhs) * cosphi  # Eq 34
        ahv = deps * mu_s * (1 + rhi) * (1 - rvs) * sinphi  # Eq 35
        avh = deps * mu_i * (1 - rvi) * (1 + rhs) * sinphi  # Eq 36
        avv = -deps * (
            mu_i * mu_s * (1 - rvi) * (1 - rvs) * cosphi - 1 / eps_r * sin_i * sin_s * (1 + rvi) * (1 + rvs)
        )  # Eq 37

        k01 = WAVENUMBER_PER_HZ * frequency * np.sqrt(eps_1).real

        # calculate W

        kl2 = (k01 * self.corr_length) ** 2
        ks2 = (k01 * self.roughness_rms) ** 2

        dkrhol2 = kl2 * ((sin_i**2 + sin_s**2) - 2 * sin_i * sin_s * cosphi)

        if self.autocorrelation_function == "gaussian":
            # gaussian C(r) = exp( - (r/l)^2 )
            Wrho = ks2 * kl2 / (4 * np.pi) * np.exp(-dkrhol2 / 4)  # eq 1.3.15 in Tsang III

        elif self.autocorrelation_function == "exponential":
            # exponential C(r) = exp( -r/l )
            # lc = self.corr_length
            raise NotImplementedError("Exponential ACF not implemented yet in Iodice SPM -- check pi normalisation")
            Wrho = TWO_PI * ks2 * kl2 * (1 + dkrhol2) ** (-1.5)
        else:
            raise SMRTError("The autocorrelation function must be exponential or gaussian")

        coef = 1 / (4 * np.pi) * Wrho / mu_i  # mu_i to convert sigma into bistatic coefficient

        coef *= np.pi  # I'm not sure why this needs to be added. From W ??

        reflection_coefficients = smrt_matrix.zeros((npol, npol, cosphi.size, mu_s.size, mu_i.size))

        reflection_coefficients[0, 0] = coef * abs2(avv)
        reflection_coefficients[0, 1] = coef * abs2(avh)
        reflection_coefficients[1, 0] = coef * abs2(ahv)
        reflection_coefficients[1, 1] = coef * abs2(ahh)

        return reflection_coefficients

    def diffuse_reflection_matrix_bragg(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol, debug=False):
        """Compute the diffuse reflection coefficients using the Bragg formulation as presented in Iodice & Imperatore
        2024.

        Coefficients are calculated for an array of incident, scattered and azimuth angles in medium 1. Medium 2 is
        where the beam is transmitted.

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
        mu_s, sin_s, mu_i, sin_i, cosphi, sinphi = vectorize_angles(mu_s, mu_i, dphi)

        sin_i2 = sin_i**2
        sin_s2 = sin_s**2

        eps_r = eps_2 / eps_1

        # Bragg coefficients

        Bhh = (
            (eps_r - 1)
            * cosphi
            / (
                (mu_i + np.sqrt(eps_r - sin_i2, dtype=np.complex128))
                * (mu_s + np.sqrt(eps_r - sin_s2, dtype=np.complex128))
            )
        )  # Eq 42

        Bhv = (
            (eps_r - 1)
            * np.sqrt(eps_r - sin_s2, dtype=np.complex128)
            * sinphi
            / (
                (mu_i + np.sqrt(eps_r - sin_i2, dtype=np.complex128))
                * (eps_r * mu_s + np.sqrt(eps_r - sin_s2, dtype=np.complex128))
            )
        )  # Eq 43

        Bvh = (
            (eps_r - 1)
            * np.sqrt(eps_r - sin_i2, dtype=np.complex128)
            * sinphi
            / (
                mu_s
                + np.sqrt(eps_r - sin_s2, dtype=np.complex128)
                * (eps_r * mu_i + np.sqrt(eps_r - sin_i2, dtype=np.complex128))
            )
        )  # Eq 44

        Bvv = (
            -(eps_r - 1)
            * (
                np.sqrt(eps_r - sin_i2, dtype=np.complex128) * np.sqrt(eps_r - sin_s2, dtype=np.complex128) * cosphi
                - (eps_r * sin_i * sin_s)
            )
            / (
                (eps_r * mu_i + np.sqrt(eps_r - sin_i2, dtype=np.complex128))
                * (eps_r * mu_s + np.sqrt(eps_r - sin_s2, dtype=np.complex128))
            )
        )  # Eq 45

        k = WAVENUMBER_PER_HZ * frequency * np.sqrt(eps_1).real

        # ux = k * (sin_i - sin_s * cosphi)  # eq 33
        # uy = k * (0 - sin_s * sinphi)  # eq 33

        # calculate W

        kl2 = (k * self.corr_length) ** 2
        ks2 = (k * self.roughness_rms) ** 2

        dkrhol2 = kl2 * ((sin_i**2 + sin_s**2) - 2 * sin_i * sin_s * cosphi)

        if self.autocorrelation_function == "gaussian":
            # gaussian C(r) = exp( - (r/l)^2 )
            Wrho = ks2 * kl2 / (4 * np.pi) * np.exp(-dkrhol2 / 4)  # eq 1.3.15 in Tsang III

        elif self.autocorrelation_function == "exponential":
            # exponential C(r) = exp( -r/l )
            # lc = self.corr_length
            raise NotImplementedError("Exponential ACF not implemented yet in Bragg SPM -- Check pi normalisation")
            Wrho = TWO_PI * ks2 * kl2 * (1 + dkrhol2) ** (-1.5)
        else:
            raise SMRTError("The autocorrelation function must be exponential or gaussian")

        # devide the sigma by mu_i to gte bistatic coefficients #coef = 4 / np.pi * mu_s**2 * mu_i** 2 * Wrho / mu_i
        coef = 4 / np.pi * mu_s**2 * mu_i * Wrho

        coef *= np.pi  # I'm not sure why this needs to be added

        reflection_coefficients = smrt_matrix.zeros((npol, npol, cosphi.size, mu_s.size, mu_i.size))

        reflection_coefficients[0, 0] = coef * abs2(Bvv)
        reflection_coefficients[0, 1] = coef * abs2(Bvh)
        reflection_coefficients[1, 0] = coef * abs2(Bhv)
        reflection_coefficients[1, 1] = coef * abs2(Bhh)

        return reflection_coefficients

    def diffuse_reflection_matrix_tsang(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol, debug=False):
        """Compute the diffuse reflection coefficients.

        Coefficients are calculated for an array of incident, scattered and azimuth angles in medium 1. Medium 2 is
        where the beam is transmitted.

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
        mu_s, sin_s, mu_i, sin_i, cosphi, sinphi = vectorize_angles(mu_s, mu_i, dphi)

        k0 = WAVENUMBER_PER_HZ * frequency
        k = k0 * np.sqrt(eps_1)
        k1 = k0 * np.sqrt(eps_2)

        krho = k * sin_s
        krhoi = k * sin_i

        kz = k * mu_s
        kzi = k * mu_i
        k1z = np.sqrt(k1**2 - krho**2)
        k1zi = np.sqrt(k1**2 - krhoi**2)

        # f1
        coef = k1**2 - k**2

        # e pol is Transvers electric = H pol !!
        # h pol is Transvers Magnetic = V pol !!

        f1ee = coef / (kz + k1z) * (2 * kzi / (kzi + k1zi)) * cosphi  # Eq 1.2.52a in Tsang III
        f1he = (
            coef * k1z * k / (k1**2 * kz + k**2 * k1z) * (2 * kzi / (kzi + k1zi)) * sinphi
        )  # Eq 1.2.52b in Tsang III.
        f1eh = coef / (kz + k1z) * kzi * 2 * k * k1zi / (k**2 * k1zi + k1**2 * kzi) * sinphi  # Eq 1.2.52c in Tsang III.
        f1hh = (
            coef
            / (k1**2 * kz + k**2 * k1z)
            * 2
            * k**2
            * kzi
            / (k1**2 * kzi + k**2 * k1zi)
            * (-k1z * k1zi * cosphi + krho * krhoi * k1**2 / k**2)
        )  # Eq 1.2.52d in Tsang III

        # calculate W
        l2 = self.corr_length**2
        s2 = self.roughness_rms**2
        delta_krho2 = krho**2 + krhoi**2 - 2 * krho * krhoi * cosphi

        if self.autocorrelation_function == "gaussian":
            Wrho = s2 * l2 / (4 * np.pi) * np.exp(-delta_krho2 * l2 / 4)  # eq 1.3.15 in Tsang III
        #
        elif self.autocorrelation_function == "exponential":
            # Exponential C(r) = exp(-r / l_c)

            # |k_rho - k_rhoi|^2 = krho^2 + krhoi^2 - 2*krho*krhoi*cosphi

            # Fourier transform of exponential ACF
            Wrho = TWO_PI * s2 * l2 / (1 + l2 * delta_krho2) ** 1.5
        else:
            raise SMRTError("The autocorrelation function must be exponential or gaussian")

        coef = k**2 * mu_s**2 / mu_i * Wrho  # Eq 1.3.42 in Tsang III

        reflection_coefficients = smrt_matrix.zeros((npol, npol, cosphi.size, mu_s.size, mu_i.size))

        reflection_coefficients[0, 0] = coef * abs2(f1hh)  # hh=VV  !!!
        reflection_coefficients[0, 1] = coef * abs2(f1he)
        reflection_coefficients[1, 0] = coef * abs2(f1eh)
        reflection_coefficients[1, 1] = coef * abs2(f1ee)  # ee=HH

        return reflection_coefficients

    def ft_even_diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol):
        def reflection_function(dphi):
            return self.diffuse_reflection_matrix(frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol=npol)

        print("to be optimised")
        return generic_ft_even_matrix(reflection_function, m_max, nsamples=256)

    def diffuse_transmission_matrix(self, frequency, eps_1, eps_2, mu_t, mu_i, dphi, npol):
        """Compute the diffuse transmission coefficients.

        Coefficient are calculated for an array of incident, scattered and azimuth angles in medium 1. Medium 2 is
        where the beam is transmitted.

        Args:
            frequency: Frequency of the incident wave.
            eps_1: Permittivity of the medium where the incident beam is propagating.
            eps_2: Permittivity of the other medium.
            mu_t: Array of cosine of transmitted wave angles.
            mu_i: Array of cosine of incident angles.
            dphi: Azimuth angles.
            npol: Number of polarization.

        Returns:
            The transmission matrix.
        """
        # See from eq 63 in Chen 2011 (Random Media)
        mu_t, sin_t, mu_i, sin_i, cosphi, sinphi = vectorize_angles(mu_t, mu_i, dphi)

        k0 = WAVENUMBER_PER_HZ * frequency
        k = k0 * np.sqrt(eps_1)  # in medium 1
        k1 = k0 * np.sqrt(eps_2)  # in medium 2

        krho = k1 * sin_t  # k1 because defined in medium 2
        krhoi = k * sin_i

        kz = np.sqrt(k**2 - krho**2, dtype=np.complex128)  # in medium 1
        kzi = np.sqrt(k**2 - krhoi**2, dtype=np.complex128)  # in medium 1

        k1z = np.sqrt(k1**2 - krho**2, dtype=np.complex128)  # in medium 2
        k1zi = np.sqrt(k1**2 - krhoi**2, dtype=np.complex128)  # in medium 2

        # f1
        coef = k1**2 - k**2

        # e pol is Transvers electric = H pol !!
        # h pol is Transvers Magnetic = V pol !!

        # compute f transmission coefficients
        # the equation for the TE pol (e-pol) are the same as those for the reflection in Tsang III
        # The equation for the TM pol (h-pol) are different from those for the reflection in Tsang III

        f1ee = (
            coef / (kz + k1z) * (2 * kzi) / (kzi + k1zi) * cosphi
        )  # Eq 43 in Chen 2011  (same as Eq 1.2.52a in Tsang III)
        f1eh = -coef * (kz * k1) / (k1**2 * kz + k**2 * k1z) * (2 * kzi) / (kzi + k1zi) * sinphi  # Eq 44

        f1he = (
            coef / (kz + k1z) * (2 * kzi * k * k1zi) / (k1**2 * kzi + k**2 * k1zi) * sinphi
        )  # Eq 45 (same as Eq 1.2.52c in Tsang III)
        f1hh = (
            coef
            / (k1**2 * kz + k**2 * k1z)
            * (2 * k * k1 * kzi)
            / (k1**2 * kzi + k**2 * k1zi)
            * (k1z * k1zi * cosphi + krho * krhoi)
        )  # Eq 46   (not same as Tsang III)

        # calculate W
        l2 = self.corr_length**2
        s2 = self.roughness_rms**2
        delta_krho2 = krho**2 + krhoi**2 - 2 * krho * krhoi * cosphi

        if self.autocorrelation_function == "gaussian":
            Wrho = s2 * l2 / (4 * np.pi) * np.exp(-delta_krho2 * l2 / 4)  # eq 1.3.15 in Tsang III
        #
        elif self.autocorrelation_function == "exponential":
            # Exponential C(r) = exp(-r / l_c)

            # |k_rho - k_rhoi|^2 = krho^2 + krhoi^2 - 2*krho*krhoi*cosphi

            # Fourier transform of exponential ACF
            Wrho = TWO_PI * s2 * l2 / (1 + l2 * delta_krho2) ** 1.5
        else:
            raise SMRTError("The autocorrelation function must be exponential or gaussian")

        coef = k1 * k1z**2 / kzi * Wrho  # from Chen 2011 eq 64--67

        transmission_coefficients = smrt_matrix.zeros((npol, npol, cosphi.size, mu_t.size, mu_i.size))

        transmission_coefficients[0, 0] = coef * abs2(f1hh)  # hh=VV  !!!
        transmission_coefficients[0, 1] = coef * abs2(f1he)
        transmission_coefficients[1, 0] = coef * abs2(f1eh)
        transmission_coefficients[1, 1] = coef * abs2(f1ee)  # ee=HH

        return transmission_coefficients

    def ft_even_diffuse_transmission_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol):
        def transmission_function(dphi):
            return self.diffuse_transmission_matrix(frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol=npol)

        print("to be optimised")
        return generic_ft_even_matrix(transmission_function, m_max, nsamples=256)
