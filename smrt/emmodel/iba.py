# coding: utf-8

"""Compute scattering from Improved Born Approximation theory as described in Mätzler 1998 and Mätzler and Wiesman 1999, except the
absorption coefficient which is computed with Polden von Staten formulation instead of the Eq 24 in Mätzler 1998. See iba_original.py for
a fully conforming IBA version.
 This model allows for different microstructural models provided that the Fourier transform of the correlation function
may be performed. All properties relate to a single layer.

"""

# Stdlib import

# other import
import numpy as np
import scipy.integrate
import scipy.fftpack

# local import
from ..core.error import SMRTError
from ..core.globalconstants import C_SPEED
from .effective_permittivity import depolarization_factors, polder_van_santen

#
# For developers: all emmodel must implement the `effective_permittivity`, `ke` and `phase` functions with the same arguments as here
# initialisation and precomputation can be done in the prepare method that is called only once for each layer whereas
# phase, ke and effective_permittivity can be called several times.
#


def derived_IBA(effective_permittivity_model=polder_van_santen):  # , absorption_calculation=None):
    """return a new IBA model with variant from the default IBA.

    :param effective_permittivity_model: permittivity mixing formula. Must be a function of 4 parameters (frac_volume, e0, es, depol_xyz).

    :returns a new class inheriting from IBA but with patched methods
    """
    new_class_name = "IBA_%s" % (effective_permittivity_model.__name__)  # , absorption_calculation)

    return type(new_class_name, (IBA, ), {'effective_permittivity_model' : staticmethod(effective_permittivity_model)})


class IBA(object):

    """
    Improved Born Approximation electromagnetic model class.

    As with all electromagnetic modules, this class is used to create an electromagnetic
    object that holds information about the effective permittivity, extinction coefficient and
    phase function for a particular snow layer. Due to the frequency dependence, information
    about the sensor is required. Passive and active sensors also have different requirements on
    the size of the phase matrix as redundant information is not calculated for the
    passive case.

    :param sensor: object containing sensor characteristics
    :param layer: object containing snow layer characteristics (single layer)


    **Usage Example:**

        This class is not normally accessed directly by the user, but forms part of the
        smrt model, together with the radiative solver (in this example, `dort`) i.e.:

        ::

            from smrt import make_model
            model = make_model("iba", "dort")

        `iba` does not need to be imported by the user due to autoimport of electromagnetic model modules

    """

    # default effective_permittivity_model is polder_van_santen in Matzler 1998 and Matzler&Wiesman 1999
    effective_permittivity_model = staticmethod(polder_van_santen)


    def __init__(self, sensor, layer):

        # Set size of phase matrix: active needs an extended phase matrix
        if sensor.mode == 'P':
            self.npol = 2
            self.m_max = 0
        else:
            self.npol = 3
            self.m_max = 3

        # Bring layer and sensor properties into emmodel
        self.frac_volume = layer.frac_volume
        self.microstructure = layer.microstructure  # Do this here, so can pass FT of correlation fn to phase function
        self.e0 = layer.permittivity(0, sensor.frequency)  # background permittivity
        self.eps = layer.permittivity(1, sensor.frequency)  # scatterer permittivity
        self.k0 = 2 * np.pi * sensor.frequency / C_SPEED  # Wavenumber in free space
        self.inclusion_shape = layer.inclusion_shape # for assuming spherical or ellipsoidal inclusions

        # Calculate depolarization factors and iba_coefficient
        self.depol_xyz = depolarization_factors()
        self._effective_permittivity = self.effective_permittivity()
        self.iba_coeff = self.compute_iba_coeff()

        # Absorption coefficient for general lossy medium under assumption of low-loss medium.
        self.ka = self.compute_ka()

        # Calculate scattering coefficient: integrate p11+p12 over mu
        k = 6  # number of samples. This should be adaptative depending on the size/wavelength
        mu = np.linspace(1, -1, 2**k + 1)
        y = self.ks_integrand(mu)
        ks_int = scipy.integrate.romb(y, mu[0] - mu[1])  # integrate between 0 and pi (i.e. mu between -1 and 1)
        self.ks = ks_int / 4.  # Ding et al. (2010), normalised by (1/4pi)

        assert(self.ks >= 0)

    def compute_iba_coeff(self):
        """ Calculate angular independent IBA coefficient: used in both scattering coefficient and phase function calculations

            .. note::

                Requires mean squared field ratio (uses mean_sq_field_ratio method)

        """
        y2 = self.mean_sq_field_ratio(self.e0, self.eps)
        iba_coeff = (1. / (4. * np.pi)) * np.absolute(self.eps - self.e0)**2. * y2 * (self.k0)**4
        return iba_coeff

    def mean_sq_field_ratio(self, e0, eps):
        """ Mean squared field ratio calculation

            Uses layer effective permittivity

            :param e0: background relative permittivity
            :param eps: scattering constituent relative permittivity

        """
        quasi_permittivity = (2. * self._effective_permittivity + e0) / 3.
        y2 = (1. / 3.) * np.sum(np.absolute(quasi_permittivity / (quasi_permittivity + (eps - e0) * self.depol_xyz))**2.)
        return y2

    def basic_check(self):
        # Need to be defined
        pass

    def set_max_mode(self, m_max):
        """ Sets the maximum level of phase matrix Fourier decomposition needed. Called by
            the radiative transfer solver.

            :param m_max: maximum Fourier decomposition mode needed

            .. note::

                m_max = 0 for passive

        """
        self.m_max = m_max

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

        k_diff = np.asarray(2. * self.k0 * sintheta_2 * abs(np.sqrt(self._effective_permittivity)))

        # Calculate microstructure term
        if hasattr(self.microstructure, 'ft_autocorrelation_function'):
            ft_corr_fn = self.microstructure.ft_autocorrelation_function(k_diff)
        else:
            raise SMRTError("Fourier Transform of this microstructure model has not been defined, or there is a problem with its calculation")

        p11 = (self.iba_coeff * ft_corr_fn).real * mu**2
        p22 = (self.iba_coeff * ft_corr_fn).real * 1.

        ks_int = (p11 + p22)

        return ks_int.real

    def ft_even_phase(self, m, mu_s, mu_i, npol=None):
        """IBA phase matrix.

        These are the Fourier decomposed phase matrices for modes m = 0, 1, 2.... This method
        creates or accesses the cache of ft_phase so Fourier Decomposition only needs to be
        done once per layer for all modes.

        Coefficients within the phase function are

        Passive case (m = 0 only) and active (m = 0) ::

            M  = [Pvvp  Pvhp]
                 [Phvp  Phhp]

        Active case (m > 0)::

            M =  [Pvvp Pvhp Pvup]
                 [Phvp Phhp Phup]
                 [Puvp Puhp Puup]

        :param m: mode for decomposed phase matrix (0, 1, 2)
        :param mu: 1-D array of cosines of incidence angle
        :param npol: [Optional] number of polarizations - normally set from sensor properties
        :returns cached_phase[m]: cached phase matrix for all scattering streams for one Fourier Decomposition mode

        """
        if npol is None:
            npol = self.npol  # npol is set from sensor mode except in call to energy conservation test

        assert mu_s is mu_i   # temporary
        mu = mu_i
        
        if (not hasattr(self, "cached_mu")) or (not np.array_equal(self.cached_mu, mu)) or (len(self.cached_phase) < m):
            self.precompute_ft_even_phase(mu, max(m, self.m_max), npol)
        return self.cached_phase[m]

    def phase(self, mu_s, mu_i, dphi, npol=2):
        """ IBA Phase function (not decomposed).

"""
        # cos and sin of scattering and incident angles in the main frame
        cos_ti = np.atleast_1d(mu_i)[np.newaxis, np.newaxis, :]
        sin_ti = np.sqrt(1. - cos_ti**2)

        cos_t = np.atleast_1d(mu_s)[np.newaxis, :, np.newaxis]
        sin_t = np.sqrt(1. - cos_t**2)

        dphi = np.atleast_1d(dphi)
        cos_pd = np.cos(dphi)[:, np.newaxis, np.newaxis]

        # Scattering angle in the 1-2 frame
        cosT = cos_t * cos_ti + sin_t * sin_ti * cos_pd
        cosT = np.clip(cosT, -1.0, 1.0)  # Prevents occasional numerical error
        cosT2 = cosT**2  # cos^2 (Theta)
        sinT = np.sqrt(1. - cosT2)

        # Create arrays of rotation angles: without the scattering angle denominator
        cos_i1 = cos_t * sin_ti - cos_ti * sin_t * cos_pd
        cos_i2 = cos_ti * sin_t - cos_t * sin_ti * cos_pd

        # Apply non-zero scattering denominator
        nonnullsinT = np.abs(sinT) >= 1e-6
        cos_i1[nonnullsinT] /= sinT[nonnullsinT]
        cos_i2[nonnullsinT] /= sinT[nonnullsinT]

        # Special condition if theta and theta_i = 0 to preserve azimuth dependency
        dege_dphi = np.broadcast_to( (np.abs(sin_t) < 1e-6) & (np.abs(sin_ti) < 1e-6), cos_i1.shape)
        cos_i1[dege_dphi] = 1.
        cos_i2[dege_dphi] = np.broadcast_to(cos_pd, cos_i2.shape)[dege_dphi]

        # Calculate rotation angles alpha, alpha_i
        # Convention follows Matzler 2006, Thermal Microwave Radiation, p111, eqn 3.20
        cosa = -np.clip(cos_i2, -1.0, 1.0)  # cos (pi - i2)
        cosai = np.clip(cos_i1, -1.0, 1.0)  # cos (-i1)

        # Calculate arrays of rotated phase matrix elements
        # Shorthand to make equations shorter & marginally faster to compute
        cosa2 = cosa**2  # cos^2 (alpha)
        cosai2 = cosai**2  # cos^2 (alpha_i)
        sina2 = 1 - cosa2  # sin^2 (alpha)
        sinai2 = 1 - cosai2  # sin^2 (alpha_i)
        sin2a = - 2 * cosa * np.sqrt(sina2)  # sin(2 alpha)
        sin2ai = 2 * cosai * np.sqrt(sinai2)  # sin(2 alpha_i)
        cos2a = 2 * cosa2 - 1  # cos(2 alpha): needed for active only
        cos2ai = 2 * cosai2 - 1  # cos(2 alpha_i): needed for active only

        # For pi < phi_diff <  2 * pi, it is necessary to change the sign of i2 and i1
        # (see Matzler 2006 pg 113.)
        # This will only affect the sin2a and sin2ai calculations: all others are cos and/or squared
        change_sign = dphi >= np.pi
        sin2a[change_sign, :, :] = -sin2a[change_sign, :, :]
        sin2ai[change_sign, :, :] = -sin2ai[change_sign, :, :]


        # IBA phase function = rayleigh phase function * angular part of microstructure term
        # Calculate wavevector difference
        k_diff = 2. * self.k0 * np.sqrt(self._effective_permittivity) * np.sqrt((1. - cosT) / 2.)
        # Calculate microstructure term
        if hasattr(self.microstructure, 'ft_autocorrelation_function'):
            ft_corr_fn = self.microstructure.ft_autocorrelation_function(k_diff)
        else:
            raise SMRTError("Fourier Transform of this microstructure model has not been defined, or there is a problem with its calculation")

        p_second_term = 0.5 * sin2a * cosT * sin2ai
        p11 = (cosa2 * cosT2 * cosai2 + sina2 * sinai2 - p_second_term)  #p11
        p12 = (cosa2 * cosT2 * sinai2 + sina2 * cosai2 + p_second_term)  #p12
        p21 = (sina2 * cosT2 * cosai2 + cosa2 * sinai2 + p_second_term)  #p21
        p22 = (sina2 * cosT2 * sinai2 + cosa2 * cosai2 - p_second_term) #p22


        if npol == 2:
            p = np.array([[p11, p12], [p21, p22]])

        elif npol == 3: # 3-pol
            p = np.array([[p11, p12, 0.5 * ( cosa2 * cosT2 * sin2ai - sina2 * sin2ai + sin2a * cosT * cos2ai)], # p13
                          [p21, p22, 0.5 * ( sina2 * cosT2 * sin2ai - cosa2 * sin2ai - sin2a * cosT * cos2ai)], # p23
                          [(-sin2a * cosT2 * cosai2 + sin2a * sinai2 - cos2a * cosT * sin2ai),      # p31
                           (-sin2a * cosT2 * sinai2 + sin2a * cosai2 + cos2a * cosT * sin2ai),      # p32
                           (-0.5 * sin2a * cosT2 * sin2ai - 0.5 * sin2a * sin2ai + cos2a * cosT * cos2ai)] # p33
                         ])
        else:
            raise RuntimeError("invalid value of npol")

        return (ft_corr_fn * self.iba_coeff * p).squeeze()


    def precompute_ft_even_phase(self, mu, m_max, npol):
        """ Calculation of the Fourier decomposed IBA phase function.

        This method calculates the Improved Born Approximation phase matrix for all
        Fourier decomposition modes and stores the output in a cache so the calculation
        is not repeated for each mode. The radiative transfer solver then accesses the cache.

        The IBA phase function is given in Mätzler, C. (1998). Improved Born approximation for
        scattering of radiation in a granular medium. *Journal of Applied Physics*, 83(11),
        6111-6117. Here, calculation of the phase matrix is based on the phase matrix in
        the 1-2 frame, which is then rotated according to the incident and scattering angles,
        as described in e.g. *Thermal Microwave Radiation: Applications for Remote Sensing, Mätzler (2006)*.
        Fourier decomposition is then performed to separate the azimuthal dependency from the incidence angle dependency.

        :param mu: 1-D array of cosine of radiation stream angles (set by solver)
        :param m_max: maximum Fourier decomposition mode needed
        :param npol: number of polarizations considered (set from sensor characteristics)

        Calculates cached_phase: Stored phase matrix for each Fourier mode m

        .. note::

            The size of the cached_phase[m] matrix depends on the mode. Only p11, p12, p21 and p22
            elements are needed for the m = 0 mode, whereas an extended matrix with the p13, p23, p31, p32 and p33
            elements are required for m > 0 modes (active only). The size of the cached phase matrix
            will also vary with snow layer, as it depends on the number of streams simulated (length of mu).

        """
        # Raise exception if mu = 1 ever called for active: p13, p23, p31, p32 signs incorrect
        if any(u == 1 for u in mu) and npol > 2:
            raise SMRTError("Phase matrix signs for sine elements of mode m = 2 incorrect")

        nsamples = 2**(m_max + 3) #* (m_max + 1)  # 2**4  # samples of dphi for fourier decomposition. Highest efficiency for 2^n. 2^2 ok
        dphi_interval = 2. * np.pi / nsamples  # sampling interval. Period is 2pi
        dphi = np.arange(0, 2. * np.pi, dphi_interval)  # evenly spaced from 0 to period (but not including period)

        # Determine size of mode-dependent array
        # 2 x 2 phase matrix for mode m=0, otherwise 3 x 3
        pm_size = ([2] + [npol] * m_max)
        self.cached_phase = [np.empty((pm_size[m] * len(mu), pm_size[m] * len(mu))) for m in range(m_max + 1)]
        self.cached_mu = mu

        p = self.phase(mu, mu, dphi, npol)
        decomposed_p = np.fft.fft(p, axis=2)

        delta = 1.0 / dphi.size  # Delta is 1 for m=0 mode
        self.cached_phase[0][0::2, 0::2] = decomposed_p[0, 0, 0].real * delta
        self.cached_phase[0][0::2, 1::2] = decomposed_p[0, 1, 0].real * delta
        self.cached_phase[0][1::2, 0::2] = decomposed_p[1, 0, 0].real * delta
        self.cached_phase[0][1::2, 1::2] = decomposed_p[1, 1, 0].real * delta

        for m in range(1, m_max + 1):
            delta = 2.0 / dphi.size  # Delta is 1 for m=0 mode
            self.cached_phase[m][0::npol, 0::npol] = decomposed_p[0, 0, m].real * delta
            self.cached_phase[m][0::npol, 1::npol] = decomposed_p[0, 1, m].real * delta
            self.cached_phase[m][1::npol, 0::npol] = decomposed_p[1, 0, m].real * delta
            self.cached_phase[m][1::npol, 1::npol] = decomposed_p[1, 1, m].real * delta

            #print(self.cached_phase[m][0::npol, 0::npol])
            if (m>0) and (npol>=3):
                # For the even matrix:
                # Sin components needed for p31, p32. Negative sin components needed for p13, p23. Cos for p33
                self.cached_phase[m][0::npol, 2::npol] = - decomposed_p[0, 2, m].imag * delta
                self.cached_phase[m][1::npol, 2::npol] = - decomposed_p[1, 2, m].imag * delta
                self.cached_phase[m][2::npol, 0::npol] = decomposed_p[2, 0, m].imag * delta
                self.cached_phase[m][2::npol, 1::npol] = decomposed_p[2, 1, m].imag * delta
                self.cached_phase[m][2::npol, 2::npol] = decomposed_p[2, 2, m].real * delta

    def compute_ka(self):
        """ IBA absorption coefficient calculated from the low-loss assumption of a general lossy medium.

        Calculates ka from wavenumber in free space (determined from sensor), and effective permittivity
        of the medium (snow layer property)

        :return ka: absorption coefficient [m :sup:`-1`]

        .. note::

            This may not be suitable for high density material

        """

        # after several go and back, the situation is now clear:
        # MEMLS uses the formulation in IBA98 paper. In SMRT this formulation is available in iba_original.py
        # here we use Polden von Staten which is known to be better and accommodate the full range of density/frac_volume
        # PvS is also now recommended by Christian Matzler and has been implemented in MEMLS modified for sea-ice.
        # This is therefore the default in SMRT. The fully MEMLS compatible IBA is in iba_original.py

        return 2 * self.k0 * np.sqrt(self._effective_permittivity).imag


    def ke(self, mu):
        """ IBA extinction coefficient matrix

        The extinction coefficient is defined as the sum of scattering and absorption
        coefficients. However, the radiative transfer solver requires this in matrix form,
        so this method is called by the solver.

            :param mu: 1-D array of cosines of radiation stream incidence angles
            :returns ke: extinction coefficient matrix [m :sup:`-1`]

            .. note::

                Spherical isotropy assumed (all elements in matrix are identical).

                Size of extinction coefficient matrix depends on number of radiation
                streams, which is set by the radiative transfer solver.

        """
        return np.full(len(mu), self.ks + self.ka)

    def effective_permittivity(self):
        """ Calculation of complex effective permittivity of the medium.

        :returns effective_permittivity: complex effective permittivity of the medium

        """

        eps = type(self).effective_permittivity_model(
            self.frac_volume, self.e0, self.eps, self.depol_xyz, self.inclusion_shape)
        
        if eps.imag < 0:
            raise SMRTError("the imaginary part of the permittivity must be positive, by convention, in SMRT")
        return eps


class IBA_MM(IBA):
    # Undocumented: this is test code for comparison with MEMLS, and may be removed from later versions.

    def __init__(self, sensor, layer):
        IBA.__init__(self, sensor, layer)  # Gives all IBA parameters. Some need to be recalculated (effective permittivity, scattering and absorption coefficients)

        self._effective_permittivity = polder_van_santen(self.frac_volume)

        # Imaginary component for effective permittivity from Wiesmann and Matzler (1999)
        y2 = self.mean_sq_field_ratio(self.e0, self.eps)
        effective_permittivity_imag = self.frac_volume * self.eps.imag * y2 * np.sqrt(self._effective_permittivity)
        self._effective_permittivity = self._effective_permittivity + 1j * effective_permittivity_imag

        self.iba_coeff = self.compute_iba_coeff()
        ks_int, ks_err = scipy.integrate.quad(self._mm_integrand, 0, np.pi)
        self.ks = ks_int / 2.  # Matzler and Wiesmann, RSE, 1999, eqn (8)
        # General lossy medium under assumption of low-loss medium.
        self.ka = self.compute_ka()

    def _mm_integrand(self, theta):
        # Calculate wavevector difference
        k_diff = np.asarray(2. * self.k0 * np.sin(theta / 2.) * np.sqrt(self._effective_permittivity))

        # Calculate microstructure term
        if hasattr(self.microstructure, 'ft_autocorrelation_function'):
            ft_corr_fn = self.microstructure.ft_autocorrelation_function(k_diff)
        else:
            raise SMRTError("Fourier Transform of this microstructure model has not been defined, or there is a problem with its calculation")

        # MEMLS phase function has mean of H and V polarisation angle. Eqn 17c of Matzler and Wiesmann 1999.
        p_mm = self.iba_coeff * ft_corr_fn.real * (1. - 0.5 * np.square(np.sin(theta)))
        ks_int = p_mm * np.sin(theta)

        return ks_int.real
