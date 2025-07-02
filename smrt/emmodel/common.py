
import inspect

import numpy as np

from smrt.core.lib import abs2, smrt_matrix, len_atleast_1d
from smrt.core.error import SMRTError


def vectorize_angles(mu_s, mu_i, dphi, compute_cross_product=True, compute_sin=True):
    """return angular cosines and sinus with proper dimensions, ready for vectorized calculations.

    Args:
        mu_s: scattering cosine angle.
        mu_i: incident cosine angle.
        dphi: azimuth angle between the scattering and incident directions.
        compute_cross_product: if True perform the computation for all elements in mu_s, mu_i, dphi (cross product)
            and if False perform the computation for each successive configuration in mu_s, mu_i and dphi (they must
            have the same shape).

    Returns:
        vectorize angles
    """

    mu_s = np.atleast_1d(mu_s)
    mu_i = np.atleast_1d(mu_i)
    dphi = np.atleast_1d(dphi)

    if compute_cross_product:
        mu_s = mu_s[np.newaxis, :, np.newaxis]
        mu_i = mu_i[np.newaxis, np.newaxis, :]
        dphi = dphi[:, np.newaxis, np.newaxis]

    sin_i = np.sqrt(1. - mu_i**2) if compute_sin else np.nan
    sin_s = np.sqrt(1. - mu_s**2) if compute_sin else np.nan

    sinphi = np.sin(dphi)
    cosphi = np.cos(dphi)

    return mu_s, sin_s, mu_i, sin_i, cosphi, sinphi


def rayleigh_scattering_matrix_and_angle_tsang00(mu_s, mu_i, dphi, npol=2):
    """compute the Rayleigh matrix and half scattering angle. Based on Tsang theory and application p271 Eq 7.2.16

"""
    mu_s, sin_s, mu_i, sin_i, cosphi, sinphi = vectorize_angles(mu_s, mu_i, dphi)

    # Tsang theory and application p127 Eq 3.2.47
    fvv = cosphi * mu_s * mu_i + sin_s * sin_i
    fhv = - sinphi * mu_i   # there is an error in Tsang book. By symmetry of the equation fvh has sin(phi_s-phi_i)
    # and fhv has sin(phi_i-phi_s)
    fhh = cosphi
    fvh = sinphi * mu_s

    p = phase_matrix_from_scattering_amplitude(fvv, fvh, fhv, fhh, npol=npol)

    # compute scattering angle

    cosT = np.clip(mu_s * mu_i + sin_s * sin_i * cosphi, -1.0, 1.0)  # Prevents occasional numerical error

    sin_half_scatt = np.sqrt(0.5 * (1 - cosT))   # compute half the scattering angle

    return p, sin_half_scatt


def phase_matrix_from_scattering_amplitude(fvv, fvh, fhv, fhh, npol=2):
    """compute the phase function according to the scattering amplitude. This follows Tsang's convention.
"""

    fvv, fvh, fhv, fhh = np.broadcast_arrays(fvv, fvh, fhv, fhh)

    if npol == 2:
        p = [[abs2(fvv), abs2(fvh)],
             [abs2(fhv), abs2(fhh)]]
    elif npol == 3:
        cfhh = np.conj(fhh)
        cfhv = np.conj(fhv)

        p = [[abs2(fvv), abs2(fvh), (np.conj(fvh) * fvv).real],
             [abs2(fhv), abs2(fhh), (cfhh * fhv).real],
             [2 * (fvv * cfhv).real, 2 * (fvh * cfhh).real, (fvv * cfhh + fvh * cfhv).real]]
    else:
        raise RuntimeError("invalid number of polarisation")

    return np.array(p)


def generic_ft_even_matrix(phase_function, m_max, nsamples=None):
    """ Calculation of the Fourier decomposed of the phase or reflection or transmission matrix provided by the function.

    This method calculates the Fourier decomposition modes and return the output.

    Coefficients within the phase function are

    Passive case (m = 0 only) and active (m = 0) ::

        M  = [Pvvp  Pvhp]
             [Phvp  Phhp]

    Active case (m > 0)::

        M =  [Pvvp Pvhp Pvup]
             [Phvp Phhp Phup]
             [Puvp Puhp Puup]

    :param phase_function: must be a function taking dphi as input. It is assumed that phi is symmetrical (it is in cos(phi))
    :param m_max: maximum Fourier decomposition mode needed

    """

    # samples of dphi for fourier decomposition. Highest efficiency for 2^n. 2^2 ok
    if nsamples is None:
        nsamples = 2**np.ceil(3 + np.log(m_max + 1) / np.log(2))

    assert nsamples > 2 * m_max

    # dphi must be evenly spaced from 0 to 2 * np.pi (but not including period), but we can use the symmetry of the phase function
    # to reduce the computation to 0 to pi (including 0 and pi) and mirroring for pi to 2*pi (excluding both)

    dphi = np.linspace(0, np.pi, int(nsamples // 2 + 1))

    # compute the phase function
    p = phase_function(dphi)

    npol = p.npol

    # mirror the phase function
    assert len(p.values.shape) == 5, f"Expect 5 dimensions, got {p.values.shape}"

    p_mirror = p.values[:, :, -2:0:-1, :, :].copy()
    if npol >= 3:
        p_mirror[0:2, 2] = -p_mirror[0:2, 2]
        p_mirror[2, 0:2] = -p_mirror[2, 0:2]

    # concatenate the two mirrored phase function
    p = np.concatenate((p.values, p_mirror), axis=2)
    assert(p.shape[2] == nsamples)

    # compute the Fourier Transform of the phase function along phi axis (axis=2)
    ft_p = np.fft.fft(p, axis=2)

    #assert np.allclose(ft_p[:, :, 0, :, :], np.sum(p, axis=2)), f"Strange ... {ft_p[:, :, 0, :, :]} {np.sum(p, axis=2)}"

    ft_even_p = smrt_matrix.empty((npol, npol, m_max + 1, p.shape[-2], p.shape[-1]))

    # m=0 mode
    ft_even_p[:, :, 0] = ft_p[:, :, 0].real * (1.0 / nsamples)

    # m>=1 modes
    if npol == 2:
        # the factor 2 comes from the change exp -> cos, i.e. exp(-ix) + exp(+ix)= 2 cos(x)
        ft_even_p[:, :, 1:] = ft_p[:, :, 1:m_max + 1].real * (2.0 / nsamples)

    else:
        delta = 2.0 / nsamples
        ft_even_p[0:2, 0:2, 1:] = ft_p[0:2, 0:2, 1:m_max + 1].real * delta

        # For the even matrix:
        # Sin components needed for p31, p32. Negative sin components needed for p13, p23. Cos for p33
        # The sign for 0:2, 2 and 2, 0:2 have been double check with Rayleigh and Mazter 2006 formulation of the Rayeligh Matrix (p111-112)
        ft_even_p[0:2, 2, 1:] = ft_p[0:2, 2, 1:m_max + 1].imag * delta
        ft_even_p[2, 0:2, 1:] = - ft_p[2, 0:2, 1:m_max + 1].imag * delta
        ft_even_p[2, 2, 1:] = ft_p[2, 2, 1:m_max + 1].real * delta

    return ft_even_p  # order is pola_s, pola_i, m, mu_s, mu_i


def extinction_matrix(sigma_V, sigma_H=None, npol=2, mu=None):
    """compute the extinction matrix from the extinction in V and in H-pol.
    If sigma_V or sigma_H are a scalar, they are expanded in a diagonal matrix provided mu is given.
    If sigma_H is None, sigma_V is used.
"""

    if np.isscalar(sigma_V):
        sigma_V = np.full(len_atleast_1d(mu), sigma_V)

    if sigma_H is None:
        sigma_H = sigma_V
    elif np.isscalar(sigma_H):
        sigma_H = np.full(len_atleast_1d(mu), sigma_H)

    if npol == 2:
        return smrt_matrix(np.array([sigma_V, sigma_H]))
    elif npol == 3:
        return smrt_matrix(np.array([sigma_V, sigma_H, 0.5 * (sigma_V + sigma_H)]))
    else:
        raise NotImplementedError("npol must be 2 or 3")


def rayleigh_scattering_matrix_and_angle_maetzler06(mu_s, mu_i, dphi, npol=2):
    """compute the Rayleigh matrix and half scattering angle. Based on Mätzler 2006 book p111.
This version is relatively slow because it uses phase matrix rotations which is unnecessarily complex for the Rayleigh phase matrix
but would be of interest for other phase matrices.

"""

    # cos and sin of scattering and incident angles in the main frame
    cos_ti = np.atleast_1d(mu_i)[np.newaxis, np.newaxis, :]
    sin_ti = np.sqrt(1. - cos_ti**2)

    cos_t = np.atleast_1d(mu_s)[np.newaxis, :, np.newaxis]
    sin_t = np.sqrt(1. - cos_t**2)

    dphi = np.atleast_1d(dphi)
    cos_pd = np.cos(dphi)[:, np.newaxis, np.newaxis]
    sin_pd_sign = np.where(dphi >= np.pi, -1, 1)[:, np.newaxis, np.newaxis]

    # Scattering angle in the 1-2 frame
    cosT = np.clip(cos_t * cos_ti + sin_t * sin_ti * cos_pd, -1.0, 1.0)  # Prevents occasional numerical error
    cosT2 = cosT**2  # cos^2 (Theta)
    sinT = np.sqrt(1. - cosT2)

    # Apply non-zero scattering denominator
    nonnullsinT = sinT >= 1e-6

    # Create arrays of rotation angles
    cost_sinti = cos_t * sin_ti
    costi_sint = cos_ti * sin_t

    cos_i1 = cost_sinti - costi_sint * cos_pd
    np.divide(cos_i1, sinT, where=nonnullsinT, out=cos_i1)
    np.clip(cos_i1, -1.0, 1.0, out=cos_i1)

    cos_i2 = costi_sint - cost_sinti * cos_pd
    np.divide(cos_i2, sinT, where=nonnullsinT, out=cos_i2)
    np.clip(cos_i2, -1.0, 1.0, out=cos_i2)

    # Special condition if theta and theta_i = 0 to preserve azimuth dependency
    dege_dphi = np.broadcast_to((sin_t < 1e-6) & (sin_ti < 1e-6), cos_i1.shape)
    cos_i1[dege_dphi] = 1.
    cos_i2[dege_dphi] = np.broadcast_to(cos_pd, cos_i2.shape)[dege_dphi]

    # # See Matzler 2006 pg 111 Eq. 3.20
    # # Calculate rotation angles alpha, alpha_i
    # # Convention follows Matzler 2006, Thermal Microwave Radiation, p111, eqn 3.20

    Li = Lmatrix(cos_i1, -sin_pd_sign, (3, npol))    # L (-i1)

    if npol == 2:
        RLi = np.array([[cosT2 * Li[0][0], cosT2 * Li[0][1]],
                        Li[1], [cosT * Li[2][0], cosT * Li[2][1]]])

    elif npol == 3:
        RLi = np.array([[cosT2 * Li[0][0], cosT2 * Li[0][1], cosT2 * Li[0][2]],
                        Li[1], [cosT * Li[2][0], cosT * Li[2][1], cosT * Li[2][2]]])
    else:
        raise RuntimeError("invalid value of npol")

    Ls = Lmatrix(-cos_i2, sin_pd_sign, (npol, 3))    # L (pi - i2)
    p = np.einsum('ij...,jk...->ik...', Ls, RLi)   # multiply the outer dimension (=polarization)

    sin_half_scatt = np.sqrt(0.5 * (1 - cosT))   # compute half the scattering angle

    return p, sin_half_scatt


def Lmatrix(cos_phi, sin_phi_sign, npol):

    # Calculate arrays of rotated phase matrix elements
    # Shorthand to make equations shorter & marginally faster to compute
    cos2_phi = cos_phi**2  # cos^2 (phi)
    sin2_phi = 1 - cos2_phi  # sin^2 (phi)

    sin_2phi = 2 * cos_phi * np.sqrt(sin2_phi)  # sin(2 phi_i)
    sin_2phi *= sin_phi_sign

    if npol == (2, 3):
        s05 = 0.5 * sin_2phi
        L = [[cos2_phi, sin2_phi, s05],
             [sin2_phi, cos2_phi, -s05]]
    elif npol == (3, 2):
        L = [[cos2_phi, sin2_phi],
             [sin2_phi, cos2_phi],
             [-sin_2phi, sin_2phi]]
    else:  # 3 pol
        s05 = 0.5 * sin_2phi
        cos_2phi = 2 * cos2_phi - 1  # cos(2 alpha)
        L = [[cos2_phi, sin2_phi, s05],
             [sin2_phi, cos2_phi, -s05],
             [-sin_2phi, sin_2phi, cos_2phi]]
    return L


# select the faster version
# the equality is not perfect when theta=0 and theta_s=0

rayleigh_scattering_matrix_and_angle = rayleigh_scattering_matrix_and_angle_tsang00


class AdjustableEffectivePermittivityMixin(object):
    """
    Mixin that allows an EM model to have the effective permittivity model defined by the user instead of by the theory of the EM Model.
The EM model must declare a default effective permittivity model.

    """

    def effective_permittivity(self):
        """ Calculation of complex effective permittivity of the medium.

            :returns effective_permittivity: complex effective permittivity of the medium

        """

        # eps = type(self).effective_permittivity_model(
        #    self.frac_volume, self.e0, self.eps, self.depol_xyz, self.inclusion_shape)

        effective_permittivity_model = type(self).effective_permittivity_model

        # inspect the signature of the effective_permittivity_model
        signature = inspect.signature(effective_permittivity_model).parameters
        args = dict(e0=self.e0, eps=self.eps, frequency=self.frequency)
        args = {k: v for k, v in args.items() if k in signature}  # filter the arguments needed by the function

        eps = type(self).effective_permittivity_model(layer_to_inject=self.layer, **args)
        if eps.imag < -1e-10:
            print(eps)
            raise SMRTError("the imaginary part of the permittivity must be positive, by convention, in SMRT")
        return eps


def derived_EMModel(base_class, effective_permittivity_model):
    """return a new IBA/SCE model with variant from the default IBA/SCE.

    :param effective_permittivity_model: permittivity mixing formula.

    :returns a new class inheriting from IBA but with patched methods
    """
    new_class_name = "%s_%s" % (base_class.__name__, effective_permittivity_model.__name__)  # , absorption_calculation)

    return type(new_class_name, (base_class, ), {'effective_permittivity_model': staticmethod(effective_permittivity_model)})


class IsotropicScatteringMixin(object):

    def ks(self, mu, npol=2):
        """ Scattering coefficient matrix

        :param mu: 1-D array of cosines of radiation stream incidence angles
        :param npol: number of polarization
        :returns ke: extinction coefficient matrix [m :sup:`-1`]

        .. note::

            Spherical isotropy assumed (all elements in matrix are identical).

            Size of extinction coefficient matrix depends on number of radiation
            streams, which is set by the radiative transfer solver.

        """

        return extinction_matrix(self._ks, mu=mu, npol=npol)

    def ke(self, mu, npol=2):
        """return the extinction coefficient matrix

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

        return extinction_matrix(self._ks + self.ka, mu=mu, npol=npol)


class GenericFTPhaseMixin(object):

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
