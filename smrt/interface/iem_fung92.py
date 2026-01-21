"""
Provide the interface boundary condition under IEM formulation provided by Fung et al. 1992.

Notes:
    It only computes the backscatter diffuse reflection as described in Fung et al. 1992, the specular reflection
    and the coherent transmission. It does not compute the full bi-static diffuse reflection and transmission.
    As a consequence it is only suitable for emissivity calculation and when single scattering is dominant, usually at low
    frequency when the medium is weakly scattering. This happends when the main mechanism is the backscatter from the
    interface attenuated by the medium.  Another case is when the rough surface is relatively smooth, the specular
    reflection is relatively strong and the energy can be scattered back by the medium (double bounce). For other
    situations, this code is not recommended.

    Additionaly, IEM is known to work for a limited range of roughness. Usually it is considered valid for ks < 3 and
    ks*kl < sqrt(eps) where k is the wavenumber, s the rms height and l the correlation length. The code print a warning
    when out of this range. There is also limitation for smooth surfaces but no warning is printed.

**Usage**::

    # rms height and corr_length values work at 10 GHz
    substrate = make_soil("iem_fung92", "dobson85_peplinski95", temperature=260,
                                            roughness_rms=1e-3,
                                            corr_length=5e-2,
                                            autocorrelation_function="exponential",
                                            moisture=moisture,
                                            clay=clay, sand=sand, drymatter=drymatter)

References:
    Fung, A.K, Zongqian, L., and Chen, K.S. (1992). Backscattering from a randomly rough dielectric surface. IEEE TRANSACTIONS ON
    GEOSCIENCE AND REMOTE SENSING, 30-2. https://doi.org/10.1109/36.134085
"""

import numpy as np

from smrt.core.error import SMRTError
from smrt.core.fresnel import (
    fresnel_coefficients,
)
from smrt.core.globalconstants import C_SPEED
from smrt.core.interface import Interface
from smrt.core.lib import abs2, smrt_matrix
from smrt.core.vector3 import vector3
from smrt.interface.interface_utils import (
    KirchoffApproximationCoherentInterfaceMixin,
)


class IEM_Fung92(
    KirchoffApproximationCoherentInterfaceMixin,
    Interface,
):
    """
    Implement a moderate rough surface model with backscatter, specular reflection and transmission only.

    It is not suitable for emissivity calculations. Use with care!

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
        "warning_handling": "print",
        "series_truncation": 10,
    }

    def check_validity(self, ks, kl, eps_r):
        # check validity
        if ks > 3:
            raise SMRTError(
                f"Warning, roughness_rms is too high for the given wavelength. Limit is ks < 3. Here ks={ks:g}"
            )

        if ks * kl > np.sqrt(eps_r):
            raise SMRTError(
                f"Warning, roughness_rms or correlation_length are too high for the given wavelength. Limit is ks * kl < sqrt(eps_r). Here ks*kl={ks * kl:g} and sqrt(eps_r)={np.sqrt(eps_r):g}"
            )

    def fresnel_coefficients(self, eps_1, eps_2, mu_i, ks, kl):
        """
        Calculate the fresnel coefficients at the angle mu_i whatever is ks and kl according to the original formulation of Fung 1992
        """

        Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_i)
        return Rv, Rh

    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol, debug=False):
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
        mu_s = np.atleast_1d(mu_s)
        mu_i = np.atleast_1d(mu_i)

        if not np.allclose(mu_s, mu_i) or not np.allclose(dphi, np.pi):
            raise NotImplementedError(
                "Only the backscattering coefficient is implemented at this stage in iem_fung92."
                "This is a very preliminary implementation"
            )

        if len(np.atleast_1d(dphi)) != 1:
            raise NotImplementedError("Only the backscattering coefficient is implemented at this stage. ")

        mu = mu_i[None, :]
        k = vector3.from_angles(2 * np.pi * frequency / C_SPEED * np.sqrt(eps_1).real, mu, 0)
        eps_r = eps_2 / eps_1

        ks = np.abs(k.norm() * self.roughness_rms)
        kl = np.abs(k.norm() * self.corr_length)

        try:
            self.check_validity(ks, kl, eps_r)
        except SMRTError as e:
            if self.warning_handling == "print":
                print(e)
            elif self.warning_handling == "nan":
                return smrt_matrix.full((npol, len(mu_i)), np.nan)

        Rv, Rh = self.fresnel_coefficients(eps_1, eps_2, mu_i, ks, kl)

        fvv = 2 * Rv / mu  # Eq 44 in Fung et al. 1992
        fhh = -2 * Rh / mu  # Eq 45 in Fung et al. 1992

        # prepare the series
        N = self.series_truncation
        n = np.arange(1, N + 1, dtype=np.float64)[:, None]

        rms2 = self.roughness_rms**2

        # Kirchoff term
        Iscalar_n = (2 * k.z) ** n * np.exp(-rms2 * k.z**2)
        Ivv_n = Iscalar_n * fvv  # Eq 82 in Fung et al. 1992
        Ihh_n = Iscalar_n * fhh

        # Complementary term
        mu2 = mu**2
        sin2 = 1 - mu2
        tan2 = sin2 / mu2
        # part of Eq 91. We don't use all the simplification because we want validity for n>1, especially not np.exp(-rms2 * k.z**2)=1
        Ivv_n += k.z**n * (sin2 / mu * (1 + Rv) ** 2 * (1 - 1 / eps_r) * (1 + tan2 / eps_r))
        Ihh_n += -(k.z**n) * (sin2 / mu * (1 + Rh) ** 2 * (eps_r - 1) / mu2)  # part of Eq 95.

        # compute the series
        rms2_over_fractorial = np.cumprod(rms2 / n)[:, None]

        # Eq 82 in Fung et al. 1992
        coef = k.norm2() / 2 * np.exp(-2 * rms2 * k.z**2)
        coef_n = rms2_over_fractorial * self.W_n(n, -2 * k.x)

        sigma_vv = coef * np.sum(coef_n * abs2(Ivv_n), axis=0)
        sigma_hh = coef * np.sum(coef_n * abs2(Ihh_n), axis=0)

        # if debug:
        #    self.sigma_vv_1 = ( 8*k.norm()2**2*rms2*abs2(Rv*mu2 + (1-mu2)*(1+Rv)**2 / 2 * (1 - 1 / eps_r)) * self.W_n(1, -2 * k.x) ).flat
        #    self.sigma_hh_1 = ( 8*k.norm()2**2*rms2*abs2(Rh*mu2) * self.W_n(1, -2 * k.x) ).flat

        reflection_coefficients = smrt_matrix.zeros((npol, len(mu_i)))
        reflection_coefficients[0] = sigma_vv / (4 * np.pi * mu_i)
        reflection_coefficients[1] = sigma_hh / (4 * np.pi * mu_i)

        return reflection_coefficients

    def W_n(self, n, k):
        if self.autocorrelation_function == "gaussian":
            # gaussian C(r) = exp ( -(r/l)**2 )
            lc = self.corr_length
            return (lc**2 / (2 * n)) * np.exp(-((k * lc) ** 2) / (4 * n))
        elif self.autocorrelation_function == "exponential":
            # exponential C(r) = exp( -r/l )
            lc = self.corr_length
            return (lc / n) ** 2 * (1 + (k * lc / n) ** 2) ** (-1.5)
        else:
            raise SMRTError("The autocorrelation function must be exponential or gaussian")

    def ft_even_diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol):
        if not np.allclose(mu_s, mu_i):
            raise NotImplementedError(
                "Only the backscattering coefficient is implemented at this stage in iem_fung92."
                "This is a very preliminary implementation"
            )

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
