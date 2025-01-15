

"""
Implement the interface boundary condition under IEM formulation provided by Fung et al. 1992. in IEEE TGRS.
Use of this code requires special attention because of two issues:

1) it only computes the backscatter diffuse reflection as described in Fung et al. 1992, the specular reflection
and the coherent transmission. It does not compute the full bi-static diffuse reflection and transmission.
As a consequence it is only suitable for emissivity calculation and when single scattering is dominant, usually at low
frequency when the medium is weakly scattering. This happends when the main mechanism is the backscatter from the
interface attenuated by the medium.  Another case is when the rough surface is relatively smooth, the specular
reflection is relatively strong and the energy can be scattered back by the medium (double bounce). For other
situations, this code is not recommended.

2) Additionaly, IEM is known to work for a limited range of roughness. Usually it is considered valid for ks < 3 and
ks*kl < sqrt(eps) where k is the wavenumber, s the rms height and l the correlation length. The code print a warning
when out of this range. There is also limitation for smooth surfaces but no warning is printed.

   **Usage example:**::

        # rms height and corr_length values work at 10 GHz
        substrate = make_soil("iem_fung92", "dobson85", temperature=260,
                                            roughness_rms=1e-3,
                                            corr_length=5e-2,
                                            autocorrelation_function="exponential",
                                            moisture=moisture,
                                            clay=clay, sand=sand, drymatter=drymatter)


"""

import numpy as np

from smrt.core.fresnel import fresnel_transmission_matrix, fresnel_reflection_matrix, fresnel_coefficients
from smrt.core.lib import smrt_matrix, abs2
from smrt.core.interface import Interface
from smrt.core.globalconstants import C_SPEED
from smrt.core.error import SMRTError
from .vector3 import vector3


class IEM_Fung92(Interface):
    """A moderate rough surface model with backscatter, specular reflection and transmission only. It is not suitable
    for emissivity calculations.Use with care!

"""
    args = ["roughness_rms", "corr_length"]
    optional_args = {"autocorrelation_function": "exponential",
                     "warning_handling": "print",
                     "series_truncation": 10}

    def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """compute the reflection coefficients for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium
        :param mu1: array of cosine of incident angles
        :param npol: number of polarization

        :return: the reflection matrix
"""
        k2 = (2 * np.pi * frequency / C_SPEED) ** 2 * abs2(eps_1)
        # Eq: 2.1.94 in Tsang 2001 Tome I
        return fresnel_reflection_matrix(eps_1, eps_2, mu1, npol) * np.exp(-4 * k2 * self.roughness_rms**2 * mu1**2)

    def check_validity(self, ks, kl, eps_r):

        # check validity
        if ks > 3:
            raise SMRTError("Warning, roughness_rms is too high for the given wavelength. Limit is ks < 3. Here ks=%g" % ks)

        if ks * kl > np.sqrt(eps_r):
            raise SMRTError("Warning, roughness_rms or correlation_length are too high for the given wavelength."
                            " Limit is ks * kl < sqrt(eps_r). Here ks*kl=%g and sqrt(eps_r)=%g" % (ks * kl, np.sqrt(eps_r)))

    def fresnel_coefficients(self, eps_1, eps_2, mu_i, ks, kl):
        """calculate the fresnel coefficients at the angle mu_i whatever is ks and kl according to the original formulation of Fung 1992"""

        Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_i)
        return Rv, Rh


    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol, debug=False):
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
            raise NotImplementedError("Only the backscattering coefficient is implemented at this stage."
                                      "This is a very preliminary implementation")

        if len(np.atleast_1d(dphi)) != 1:
            raise NotImplementedError("Only the backscattering coefficient is implemented at this stage. ")

        mu = mu_i[None, :]
        k = vector3.from_angles(2 * np.pi * frequency / C_SPEED * np.sqrt(eps_1).real, mu, 0)
        eps_r = eps_2 / eps_1

        ks = np.abs(k.norm * self.roughness_rms)
        kl = np.abs(k.norm * self.corr_length)

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
        Iscalar_n = (2 * k.z)**n * np.exp(-rms2 * k.z**2)
        Ivv_n = Iscalar_n * fvv  # Eq 82 in Fung et al. 1992
        Ihh_n = Iscalar_n * fhh

        # Complementary term
        mu2 = mu**2
        sin2 = 1 - mu2
        tan2 = sin2 / mu2
        # part of Eq 91. We don't use all the simplification because we want validity for n>1, especially not np.exp(-rms2 * k.z**2)=1
        Ivv_n += k.z**n * (sin2 / mu * (1 + Rv)**2 * (1 - 1 / eps_r) * (1 + tan2 / eps_r))
        Ihh_n += -k.z**n * (sin2 / mu * (1 + Rh)**2 * (eps_r - 1) / mu2)  # part of Eq 95.

        # compute the series
        rms2_over_fractorial = np.cumprod(rms2 / n)[:, None]

        # Eq 82 in Fung et al. 1992
        coef = k.norm2 / 2 * np.exp(-2 * rms2 * k.z**2)
        coef_n = rms2_over_fractorial * self.W_n(n, -2 * k.x)

        sigma_vv = coef * np.sum(coef_n * abs2(Ivv_n), axis=0)
        sigma_hh = coef * np.sum(coef_n * abs2(Ihh_n), axis=0)

        # if debug:
        #    self.sigma_vv_1 = ( 8*k.norm2**2*rms2*abs2(Rv*mu2 + (1-mu2)*(1+Rv)**2 / 2 * (1 - 1 / eps_r)) * self.W_n(1, -2 * k.x) ).flat
        #    self.sigma_hh_1 = ( 8*k.norm2**2*rms2*abs2(Rh*mu2) * self.W_n(1, -2 * k.x) ).flat

        reflection_coefficients = smrt_matrix.zeros((npol, len(mu_i)))
        reflection_coefficients[0] = sigma_vv / (4 * np.pi * mu_i)
        reflection_coefficients[1] = sigma_hh / (4 * np.pi * mu_i)

        return reflection_coefficients

    def W_n(self, n, k):

        if self.autocorrelation_function == "gaussian":

            # gaussian C(r) = exp ( -(r/l)**2 )
            lc = self.corr_length
            return (lc**2 / (2 * n)) * np.exp(-(k * lc)**2 / (4 * n))
        elif self.autocorrelation_function == "exponential":
            # exponential C(r) = exp( -r/l )
            lc = self.corr_length
            return (lc / n)**2 * (1 + (k * lc / n)**2)**(-1.5)
        else:
            raise SMRTError("The autocorrelation function must be expoential or gaussian")

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
        """compute the transmission coefficients for the azimuthal mode m
           and for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium
        :param mu1: array of cosine of incident angles
        :param npol: number of polarization

        :return: the transmission matrix
"""
        k0 = 2 * np.pi * frequency / C_SPEED

        k_iz = k0 * np.sqrt(eps_1).real * mu1
        k_sz = k0 * np.sqrt(eps_2 - (1 - mu1**2) * eps_1).real

        return fresnel_transmission_matrix(eps_1, eps_2, mu1, npol) * np.exp(- (k_sz - k_iz)**2 * self.roughness_rms**2)
