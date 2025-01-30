""" This is an abstract base class for the Strong Expansion theory and its various approximations. It is not to be used by the end-user."""

# Stdlib import

# other import
import numpy as np

# local import
from ..core.globalconstants import C_SPEED
from ..core.error import SMRTError
from ..core.lib import smrt_matrix, generic_ft_even_matrix
from .common import rayleigh_scattering_matrix_and_angle, extinction_matrix

import scipy.integrate


class SCEBase(object):

    def __init__(self, sensor, layer, local=False, symmetrical=False, scaled=True):

        # set value of interest in self and that's it. No calculation is involved here.

        # Set size of phase matrix: active needs an extended phase matrix
        if sensor.mode == 'P':
            self.npol = 2
        else:
            self.npol = 3

        # Bring layer and sensor properties into emmodel
        self.layer = layer
        self.frac_volume = layer.frac_volume
        self.microstructure = layer.microstructure  # Do this here, so can pass FT of correlation fn to phase function
        self.e0 = layer.permittivity(0, sensor.frequency)  # background permittivity
        self.eps = layer.permittivity(1, sensor.frequency)  # scatterer permittivity
        self.frequency = sensor.frequency
        self.k0 = 2 * np.pi * sensor.frequency / C_SPEED  # Wavenumber in free space

        self.k1 = self.k0 * np.sqrt(self.e0)
        self.k2 = self.k0 * np.sqrt(self.eps)

        self._effective_permittivity = self.effective_permittivity()

        self.symmetrical = symmetrical
        self.local = local
        self.scaled = scaled

        if self.symmetrical:
            self.A2A2inv = self.compute_A2A2inv()
            self._ke, self.ks = self.compute_ke_ks_symmetrical()
        else:
            if self.scaled:
                eps_HS = permittivity_hashin_shtrikman(self.frac_volume, self.e0, self.eps)
                k_eff = self.k0 * np.sqrt(eps_HS)
            else:
                k_eff = self.k1

            self.A2 = self.compute_A2(k_eff, self.microstructure)
            self._ke, self.ks = self.compute_ke_ks()

        self.ka = self.compute_ka()

    def compute_A2(self, Q, microstructure):

        if self.local:
            return compute_A2_local(Q, microstructure)
        else:
            return compute_A2_nonlocal(Q, microstructure)

    def compute_A2A2inv(self):
        """ Compute A2 using equation 26

        """

        assert self.symmetrical

        # invert the microstructure
        inverted_microstructure = self.microstructure.inverted_medium()

        if self.scaled:
            #eps_HS = permittivity_hashin_shtrikman(self.frac_volume), self.e0, self.eps
            #eps_HS_inv = permittivity_hashin_shtrikman(self.eps, self.e0, 1 - self.frac_volume)
            #eps_symHS = (1 - self.frac_volume) * eps_HS + self.frac_volume * eps_HS_inv
            #
            #k_symHS = self.k0 * np.sqrt(eps_symHS)

            k_symHS = self.k0 * np.sqrt(self._effective_permittivity)

            A2 = self.compute_A2(k_symHS, self.microstructure)
            A2inv = self.compute_A2(k_symHS, inverted_microstructure)
        else:
            A2 = self.compute_A2(self.k1.real, self.microstructure)
            A2inv = self.compute_A2(self.k2.real, inverted_microstructure)

        return A2, A2inv

    def compute_ke_ks(self):

        assert not self.symmetrical

        # equation 67
        f = self.frac_volume
        beta = (self.eps - self.e0) / (self.eps + 2 * self.e0)

        Eeff = self.e0 * (1 + 3 * beta * f**2 / (f * (1 - beta * f) - beta * self.A2))
        Eeff0 = self.e0 * (1 + 3 * beta * f**2 / (f * (1 - beta * f)))

        ke = 2 * self.k0 * np.sqrt(Eeff).imag
        ks = ke - 2 * self.k0 * np.sqrt(Eeff0).imag

        return ke, ks

    def compute_ke_ks_symmetrical(self):

        assert self.symmetrical
        # equation D2
        A2, A2inv = self.A2A2inv

        if self.frac_volume == 0 or (self.frac_volume == 1):
            grandA2 = 2  # no scattering
            # we could directly return Eeff=e0 or eps depending on frac_volume.
        else:
            grandA2 = 2 + A2 / self.frac_volume + A2inv / (1 - self.frac_volume)

        sum_eps = self.e0 + self.eps
        prod_eps = self.e0 * self.eps
        weighted_mean_eps = self.e0 * self.frac_volume + self.eps * (1 - self.frac_volume)

        #Eeff = sum_eps / 2 \
        #    + 1 / (2 * grandA2) * (-3 * weighted_mean_eps
        #                           + np.sqrt(4 * grandA2 * (3 - grandA2) * prod_eps + (sum_eps * grandA2 - 3 * weighted_mean_eps)**2))

        delta = 4 * grandA2 * (3 - grandA2) * prod_eps + (sum_eps * grandA2 - 3 * weighted_mean_eps)**2

        Eeff = sum_eps / 2 + 1 / (2 * grandA2) * (-3 * weighted_mean_eps + np.sqrt(delta))

        delta0 = 8 * prod_eps + (sum_eps * 2 - 3 * weighted_mean_eps)**2  # same with grandA2=2

        Eeff0 = sum_eps / 2 + 1 / 4 * (-3 * weighted_mean_eps + np.sqrt(delta0))
        # Eeff0 is the same as PvS... we do a lot of calculus here.

        ke = 2 * self.k0 * np.sqrt(Eeff).imag
        ks = ke - 2 * self.k0 * np.sqrt(Eeff0).imag

        return ke, ks

    def compute_phase_norm(self):
        """Compute the norm needed for the IBA phase matrix (=Rayleigh x microstructure ) when ks is known"""

        if self.ks == 0:
            return 0

        k = 6  # number of samples. This should be adaptative depending on the size/wavelength
        mu = np.linspace(1, -1, 2**k + 1)
        y = self.ks_integrand(mu)
        ks_int = scipy.integrate.romb(y, mu[0] - mu[1])  # integrate on mu between -1 and 1

        if ks_int == 0:
            return 0

        return self.ks / (ks_int / 4.)  # Ding et al. (2010), normalised by (1/4pi)

    def ks_integrand(self, mu):
        """ This is the scattering function for the IBA model.

        It uses the phase matrix in the 1-2 frame. With incident angle chosen to be 0, the scattering
        angle becomes the scattering zenith angle:

        .. math::

            \\Theta = \\theta


        Scattering coefficient is determined by integration over the scattering angle (0 to \\pi)

        :param mu: cosine of the scattering angle (single angle)

        .. math::

            ks\\_int = p11 + p22

        The integration is performed outside this method.

        """

        # Set up scattering geometry for 1-2 frame
        # Choose incident zenith angle to be 0 so scattering angle = scattering zenith angle (use mhu)
        # phi in the 1-2 frame for calculation of p11 is pi
        # phi in the 1-2 frame for calculation of p22 is pi / 2
        # Calculate wavevector difference
        sintheta_2 = np.sqrt((1. - mu) / 2.)  # = np.sin(theta / 2.)

        k_diff = np.asarray(2. * self.k0 * sintheta_2 * np.abs(np.sqrt(self._effective_permittivity)))

        # Calculate microstructure term
        if hasattr(self.microstructure, 'ft_autocorrelation_function'):
            ft_corr_fn = self.microstructure.ft_autocorrelation_function(k_diff)
        else:
            raise SMRTError("Fourier Transform of this microstructure model has not been defined, or there is "
                            "a problem with its calculation")

        p11 = ft_corr_fn.real * mu**2
        p22 = ft_corr_fn.real * 1.

        ks_int = (p11 + p22)

        return ks_int.real

    def phase(self, mu_s, mu_i, dphi, npol=2):
        """ IBA Phase function (not decomposed).

"""

        if not hasattr(self, "_phase_norm"):
            self._phase_norm = self.compute_phase_norm()

        p, sin_half_scatt = rayleigh_scattering_matrix_and_angle(mu_s, mu_i, dphi, npol)

        # IBA phase function = rayleigh phase function * angular part of microstructure term
        k_diff = 2. * self.k0 * np.sqrt(self._effective_permittivity) * sin_half_scatt

        # Calculate microstructure term
        if hasattr(self.microstructure, 'ft_autocorrelation_function'):
            ft_corr_fn = self.microstructure.ft_autocorrelation_function(k_diff)
        else:
            raise SMRTError("Fourier Transform of this microstructure model has not been defined, or there is a "
                            "problem with its calculation")

        return smrt_matrix(self._phase_norm * ft_corr_fn * p)

    def ft_even_phase(self, mu_s, mu_i, m_max, npol=None):
        """ Calculation of the Fourier decomposed IBA phase function.

        This method calculates the Improved Born Approximation phase matrix for all
        Fourier decomposition modes and return the output.

        Coefficients within the phase function are

        Passive case (m = 0 only) and active (m = 0) ::

            M  = [Pvvp  Pvhp]
                 [Phvp  Phhp]

        Active case (m > 0)::

            M =  [Pvvp Pvhp Pvup]
                 [Phvp Phhp Phup]
                 [Puvp Puhp Puup]


        The IBA phase function is given in Mätzler, C. (1998). Improved Born approximation for
        scattering of radiation in a granular medium. *Journal of Applied Physics*, 83(11),
        6111-6117. Here, calculation of the phase matrix is based on the phase matrix in
        the 1-2 frame, which is then rotated according to the incident and scattering angles,
        as described in e.g. *Thermal Microwave Radiation: Applications for Remote Sensing, Mätzler (2006)*.
        Fourier decomposition is then performed to separate the azimuthal dependency from the incidence angle dependency.

        :param mu_s: 1-D array of cosine of viewing radiation stream angles (set by solver)
        :param mu_i: 1-D array of cosine of incident radiation stream angles (set by solver)
        :param m_max: maximum Fourier decomposition mode needed
        :param npol: number of polarizations considered (set from sensor characteristics)

        """

        if npol is None:
            npol = self.npol  # npol is set from sensor mode except in call to energy conservation test

        # Raise exception if mu = 1 ever called for active: p13, p23, p31, p32 signs incorrect
        if np.any(mu_i == 1) and npol > 2:
            raise SMRTError("Phase matrix signs for sine elements of mode m = 2 incorrect")

        # compute the phase function
        def phase_function(dphi):
            return self.phase(mu_s, mu_i, dphi, npol)

        return generic_ft_even_matrix(phase_function, m_max)  # order is pola_s, pola_i, m, mu_s, mu_i

    def compute_ka(self):
        """ SCE absorption coefficient calculated from the low-loss assumption of a general lossy medium.

        Calculates ka from wavenumber in free space (determined from sensor), and effective permittivity
        of the medium.

        :return ka: absorption coefficient [m :sup:`-1`]

        .. note::

            This may not be suitable for high density material

        """

        return 2 * self.k0 * np.sqrt(self._effective_permittivity).imag

    def ke(self, mu, npol=2):
        """ SCE extinction coefficient matrix

        The extinction coefficient is defined as the sum of scattering and absorption
        coefficients. However, the radiative transfer solver requires this in matrix form,
        so this method is called by the solver.

            :param mu: 1-D array of cosines of radiation stream incidence angles
            :param npol: number of polarizations
            :returns ke: extinction coefficient matrix [m :sup:`-1`]

            .. note::

                Spherical isotropy assumed (all elements in matrix are identical).

                Size of extinction coefficient matrix depends on number of radiation
                streams, which is set by the radiative transfer solver.

        """
        return extinction_matrix(self.ks + self.ka, mu=mu, npol=npol)


def compute_A2_local(Q, microstructure):
    # Q is the wavenumber
    # this is the short range version to compute A2. It is similar to Rechtsman08.

    p = 12
    n = 2**p   # number of samples. This should be adaptative depending on the size/wavelength

    # grid resolution, fraction of the unique characteristic scale we have
    maxr = 2**(p // 4) * microstructure.inv_slope_at_origin
    r = np.linspace(0, maxr, n + 1)

    y = r * microstructure.autocorrelation_function(r)
    integrale1 = scipy.integrate.romb(y.real, maxr / n)

    A2 = 2 * Q**2 * (integrale1 + 1j / (4 * np.pi) * microstructure.ft_autocorrelation_function(0) * Q)

    return A2


def compute_A2_nonlocal(Q, microstructure):

    margin = 4  # this number must be higher for small grains, because
    # the FT of gamma is wider, so is the integrand of ReF. It means that using the long range version
    # is not recommanded for small grains
    # this value should be adaptive

    maxq = margin * Q

    k = 12  # number of samples. This should be adaptive depending on the size/wavelength
    n = 2**k
    nQ = n // margin
    q = np.linspace(0, maxq, n + 1)

    assert q[nQ] == Q

    # start with the imaginary part - Eq (70)
    y = 2 * q * microstructure.ft_autocorrelation_function(2 * q)

    # integrate from 0 to 2Q (for Q in 0 to qmax)
    # take the real part to avoid warnings... but this remains to be explored
    primitive = scipy.integrate.cumulative_trapezoid(y.real, 2 * q.real, initial=0)

    ImF = - 1 / (2 * (2 * np.pi)**1.5) * q * primitive

    # continue with the real part. Eq (71) is much more difficult to compute than the imaginary part
    # because of the principal value, but Torquato 2021, suppmat gives a hint with Eq S111.
    # here: M = maxq

    with np.errstate(invalid='ignore'):
        y1 = ImF / ((Q + q) * q)
        y1[0] = 0  # remove the singularity in 0

        y2 = (ImF - ImF[nQ]) / (Q**2 - q**2)

        y2[nQ] = (y2[nQ - 1] + y2[nQ + 1]) / 2   # remove the singularity in Q

    y = y1 + y2

    asymptotic_integral = (ImF[nQ] - Q / maxq * ImF[-1]) * np.log(np.abs((maxq + Q) / (maxq - Q)))

    ReF = - 2 / np.pi * Q * scipy.integrate.romb(y.real, maxq / n) - 1 / np.pi * asymptotic_integral

    gamma_3_2 = 0.5 * np.sqrt(np.pi)
    A2 = - (2 * np.pi) / (2**1.5 * gamma_3_2) * (ReF + 1j * ImF[nQ])  # the factor is from eq 67

    # check ImF(Q)
    # q = np.linspace(0, 2 * Q, n + 1)
    # y = q * microstructure.ft_autocorrelation_funscipyction(q)
    # ImFQ_direct = - 1 / (2 * (2 * np.pi)**1.5) * Q * scipy.integrate.romb(y, 2 * Q / n)
    # print(ImF[nQ], ImFQ_direct)

    return A2


def permittivity_hashin_shtrikman(frac_volume, e0, eps):

    # Eq 72 in TK21
    # in fact this is the same as Maxwell-Garnett equation. In principle we don't need to redefinie it.

    beta = (eps - e0) / (eps + 2 * e0)
    eps_HS = e0 * (1 + 3 * frac_volume * beta / (1 - frac_volume * beta))

    return eps_HS
