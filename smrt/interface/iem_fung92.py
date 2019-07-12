

"""
Implement the interface boundary condition under IEM formulation provided by Fung et al. 1992. in IEEE TGRS

"""

import numpy as np

#from altim.interface.ofresnel import fresnel_transmission_matrix, fresnel_coefficients  # a modifier quand on fusionne
from smrt.core.fresnel import fresnel_transmission_matrix, fresnel_reflection_matrix, fresnel_coefficients  # a modifier quand on fusionne
from smrt.core.lib import smrt_matrix
from smrt.core.interface import Interface
from smrt.core.globalconstants import C_SPEED
from smrt.core.error import SMRTError
from .vector3 import vector3



class IEM_Fung92(Interface):
    """A moderate rough surface.

"""
    args = ["roughness_rms", ]
    optional_args = {"exponential_corr_length": 0, "gaussian_corr_length": 0}


    def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """compute the reflection coefficients for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium
        :param mu1: array of cosine of incident angles
        :param npol: number of polarization

        :return: the reflection matrix
"""
        print("inseert expo")
        return fresnel_reflection_matrix(eps_1, eps_2, mu1, npol)


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
            raise NotImplementedError("Only the backscattering coefficient is implemented at this stage. This is a very preliminary implementation")

        if len(np.atleast_1d(dphi)) != 1:
            raise NotImplementedError("Only the backscattering coefficient is implemented at this stage. ")

        mu = mu_i[None, :]
        k = vector3.from_angles(2 * np.pi * frequency / C_SPEED * np.sqrt(eps_1), mu, 0)
        eps_r = eps_2 / eps_1

        # check validity
        ks = abs(k.norm * self.roughness_rms)
        if ks > 3:
            print("Warning, roughness_rms is too high for the given wavelength. Limit is ks < 3. Here ks=", ks)

        kskl = abs(ks * k.norm * max(self.exponential_corr_length, self.gaussian_corr_length))
        if kskl > np.sqrt(eps_r):
            print("Warning, roughness_rms or correlation_length are too high for the given wavelength. Limit is ks * kl < sqrt(eps_r). Here ks=", kskl)            



        Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_i)

        fvv = 2 * Rv / mu  # Eq 44 in Fung et al. 1992
        fhh = -2 * Rh / mu  # Eq 45 in Fung et al. 1992

        # prepare the series
        N = 4
        n  = np.arange(1, N + 1)[:, None]

        rms2 = self.roughness_rms**2

        # Kirchoff term
        Iscalar_n = (2*k.z)**n * np.exp(-rms2*k.z**2)
        Ivv_n = Iscalar_n * fvv  # Eq 82 in Fung et al. 1992
        Ihh_n = Iscalar_n * fhh

        # Complementary term
        mu2 = mu**2
        sin2 = 1 - mu2
        tan2 = sin2 / mu2
        Ivv_n += k.z**n * (sin2/mu * (1 + Rv)**2 * (1 - 1 / eps_r) * (1 + tan2 / eps_r))  # part of Eq 91. We don't use all the simplification because we want validity for n>1, especially not np.exp(-rms2*k.z**2)=1
        Ihh_n += -k.z**n * (sin2/mu * (1 + Rh)**2 * (eps_r - 1) / mu2)  # part of Eq 95.

        # compute the series
        n_fractorial = np.cumprod(n)[:, None]

        # Eq 82 in Fung et al. 1992
        coef = k.norm2 / 2 * np.exp(-2*rms2*k.z**2)
        coef_n = (rms2**n / n_fractorial) * self.W_n(n, -2 * k.x)
        #print((rms2**n / n_fractorial) , abs2(Ivv_n) )

        sigma_vv = coef * np.sum(coef_n * abs2(Ivv_n) , axis=0)  
        sigma_hh = coef * np.sum(coef_n * abs2(Ihh_n) , axis=0)

        if debug:
            self.sigma_vv_1 = ( 8*k.norm2**2*rms2*abs2(Rv*mu2 + (1-mu2)*(1+Rv)**2 / 2 * (1 - 1 / eps_r)) * self.W_n(1, -2 * k.x) ).flat
            self.sigma_hh_1 = ( 8*k.norm2**2*rms2*abs2(Rh*mu2) * self.W_n(1, -2 * k.x) ).flat

        gamma_vv = sigma_vv / (4 * np.pi * mu_i)
        gamma_hh = sigma_hh / (4 * np.pi * mu_i)

        reflection_coefficients = smrt_matrix.zeros((npol, len(mu_i)))
        reflection_coefficients[0] = gamma_vv
        reflection_coefficients[1] = gamma_hh

        return reflection_coefficients

    def W_n(self, n, k):

        if self.exponential_corr_length is 0:
            if self.gaussian_corr_length is 0:
                raise SMRTError("Either exponential_corr_length or gaussian_corr_length must be set")

           # gaussian C(r) = exp ( -(r/l)**2 )
            l = self.gaussian_corr_length
            return (l**2/(2*n)) * np.exp(-(k*l)**2/(4*n))
        else:
            # exponential C(r) = exp( -r/l )
            l = self.exponential_corr_length
            return (l/n)**2 * (1 + (k*l/n)**2)**(-1.5)

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
            coef /= (m_max + 0.5) # ad hoc normalization to get the right backscatter. This is a trick to deal with the dirac.

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
        print("inseert expo")
        return fresnel_transmission_matrix(eps_1, eps_2, mu1, npol)


def abs2(c):
    return c.real**2 + c.imag**2
