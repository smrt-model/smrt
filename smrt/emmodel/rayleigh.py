# coding: utf-8

""" Compute Rayleigh scattering. This theory requires the scatterers to be smaller than the wavelength and 
the medium to be sparsely populated (eq. very low density in the case of snow).

This model is only compatible with the Independent Sphere microstructure model

"""

import numpy as np

from ..core.error import SMRTError
from ..core.globalconstants import C_SPEED
from ..core.lib import smrt_matrix, len_atleast_1d
from .common import rayleigh_scattering_matrix_and_angle, extinction_matrix


class Rayleigh(object):
    """
    """

    def __init__(self, sensor, layer):

        super().__init__()

        # check here the limit of the Rayleigh model

        f = layer.frac_volume

        e0 = layer.permittivity(0, sensor.frequency)  # background permittivity
        eps = layer.permittivity(1, sensor.frequency)  # scatterer permittivity

        self._effective_permittivity = e0  # Rayleigh is for sparse medium only

        # TODO Ghi: solve the problem of dielectric constant dependency. Which object is responsible of running
        # Probably the emmodule

        lmda = C_SPEED / sensor.frequency

        radius = layer.microstructure.radius

        k0 = 2 * np.pi / lmda

        self.ks = f * 2 * abs((eps - e0) / (eps + 2 * e0))**2 * radius**3 * e0**2 * k0**4
        self.ka = f * 9 * k0 * eps.imag * abs(e0 / (eps + 2 * e0))**2 + (1 - f) * e0.imag * k0

    def basic_check(self):
        # TODO Ghi: check the microstructure model is compatible.
        # if we want to be strict, only IndependentShpere should be valid, but in practice any
        # model of sphere with a radius can make it!
        if not hasattr(self.layer.microstructure, "radius"):
            raise SMRTError("Only microstructure_model which defined a `radius` can be used with Rayleigh scattering")


    def ft_even_phase_baseonUlaby(self, mu_s, mu_i, m_max, npol=None):
        """#
        # Equations are from pg 1188-1189 Ulaby, Moore, Fung. Microwave Remote Sensing Vol III.
        # See also pg 157 of Tsang, Kong and Shin: Theory of Microwave Remote Sensing (1985) - can be used to derive
        # the Ulaby equations.

        """

        if npol is None:
            npol = 2 if m_max == 0 else 3

        P = smrt_matrix.empty((npol, npol, m_max + 1, len(mu_s), len(mu_i)))

        mu_i2 = mu_i**2
        mu_s2 = mu_s**2

        v, h, u = 0, 1, 2

        # mode m == 0
        P[v, v, 0] = 0.5 * np.outer(mu_s2, mu_i2) + np.outer(1 - mu_s2, 1 - mu_i2)
        P[v, h, 0] = 0.5 * mu_s2[:, np.newaxis]  # mu2[:, np.newaxis]  # equiv np.dot(mu2, np.ones_like(mu2.T))
        if npol >= 3:
            P[v, u] = 0

        P[h, v, 0] = 0.5 * mu_i2[np.newaxis, :]
        P[h, h, 0] = 0.5
        if npol >= 3:
            P[h, u, 0] = 0

        if npol >= 3:
            P[u, v, 0] = 0
            P[u, h, 0] = 0
            P[u, u, 0] = 0

        if m_max >= 1:
            sint_s = np.sqrt(1. - mu_s2)
            sint_i = np.sqrt(1. - mu_i2)
            cossint_s = mu_s * sint_s
            cossint_i = mu_i * sint_i

            P[v, v, 1] = 2 * np.outer(cossint_s, cossint_i)
            P[v, h, 1] = 0
            P[v, u, 1] = np.outer(cossint_s, sint_i)

            P[h, v, 1] = 0
            P[h, h, 1] = 0
            P[h, u, 1] = 0

            P[u, v, 1] = -2 * np.outer(sint_s, cossint_i)

            P[u, h, 1] = 0
            P[u, u, 1] = np.outer(sint_s, sint_i)

        if m_max >= 2:
            P[v, v, 2] = 0.5 * np.outer(mu_s2, mu_i2)
            P[v, h, 2] = -0.5 * mu_s2[:, np.newaxis]
            P[v, u, 2] = 0.5 * np.outer(mu_s2, mu_i)

            P[h, v, 2] = -0.5 * mu_i2[np.newaxis, :]
            P[h, h, 2] = 0.5
            P[h, u, 2] = -0.5 * mu_i[np.newaxis, :]

            P[u, v, 2] = - np.outer(mu_s, mu_i2)
            P[u, h, 2] = mu_s[:, np.newaxis]
            P[u, u, 2] = np.outer(mu_s, mu_i)

        if m_max >= 3:
            P[:, :, 3:, :, :] = 0

        if npol == 3:
            P[v, u, :] = -P[v, u, :]  # minus comes from even phase function
            P[h, u, :] = -P[h, u, :]  # minus comes from even phase function

        # this normalisation is compatible with the 1/4pi normalisation used for the RT equation.
        coef = 3 * self.ks / 2   # no*fo^2 / Ks (see TsangI 3.2.49)

        return P * coef

    def ft_even_phase_basedonJin(self, mu_s, mu_i, m_max, npol=None):
        """Rayleigh phase matrix. These are the Fourier decomposed phase matrices for modes m = 0, 1, 2. It is based on
        Y.Q. Jin

        Coefficients within the phase function are::

            M  = [Pvvp  Pvhp]
                 [Phvp  Phhp]

        :param m: mode for decomposed phase matrix (0, 1, 2)
        :param mu: vector of cosines of incidence angle
        :returns: the phase matrix

        """
        assert mu_s is mu_i  # temporary hack, to be propagated
        mu = mu_i

        if npol is None:
            npol = 2 if m_max == 0 else 3

        P = smrt_matrix.empty((npol, npol, m_max + 1, len(mu_s), len(mu_i)))

        mu2 = mu**2

        v, h, u = 0, 1, 2

        # mode = 0
        P[v, v, 0] = 0.5 * np.outer(mu2, mu2) + np.outer(1 - mu2, 1 - mu2)
        P[v, h, 0] = 0.5 * mu2[:, np.newaxis]  # equiv np.dot(mu2, np.ones_like(mu2.T))
        if npol > 3:
            P[v, u, 0] = 0

        P[h, v, 0] = P[v, h, 0].T
        P[h, h, 0] = 0.5
        if npol > 3:
            P[h, u, 0] = 0

        if npol > 3: 
            P[u, v, 0] = 0
            P[u, h, 0] = 0
            P[u, u, 0] = 0

        if m_max >= 1:
            sint = np.sqrt(1. - mu2)
            cossint = mu * sint

            P[v, v, 1] = 2 * np.outer(cossint, cossint)
            P[v, h, 1] = 0
            P[v, u, 1] = np.outer(cossint, sint)

            P[h, v, 1] = 0
            P[h, h, 1] = 0
            P[h, u, 1] = 0

            P[u, v, 1] = -2 * P[v, u, 1].T  # could transpose Pvu
            P[u, h, 1] = 0
            P[u, u, 1] = np.outer(sint, sint)

        if m_max >= 2:
            P[v, v, 2] = 0.5 * np.outer(mu2, mu2)
            P[v, h, 2] = -0.5 * mu2[:, np.newaxis]  # equiv - np.dot(mu2, np.ones_like(mu2.T))
            P[v, u, 2] = 0.5 * np.outer(mu2, mu)

            P[h, v, 2] = P[v, h, 2].T
            P[h, h, 2] = 0.5
            P[h, u, 2] = -0.5 * mu[np.newaxis, :]  ## error of theta_i theta_s in Y.Q. Jin

            P[u, v, 2] = - np.outer(mu, mu2)
            P[u, h, 2] = mu[:, np.newaxis]
            P[u, u, 2] = -np.outer(mu, mu)
            raise Exception("Tsang is wrong, to be check in Y.Q. Jin again")
            P[u, u, 2] = 0   # error in Y.Q. Jin ?? According to Tsang this term is null

        if m_max >=3:
            P[:, :, 3:, :, :] = 0

        if npol == 3:
            P[v, u, :] = -P[v, u, :]  # minus comes from even phase function
            P[h, u, :] = -P[h, u, :]  # minus comes from even phase function

        # this normalisation is compatible with the 1/4pi normalisation used for the RT equation.
        coef = 3 * self.ks / 2   # no*fo^2 / Ks (see TsangI 3.2.49)

        return P * coef


    def ft_even_phase_tsang(self, mu_s, mu_i, m_max, npol=None):
        """Rayleigh phase matrix. These are the Fourier decomposed phase matrices for modes m = 0, 1, 2.
        Equations are from p128 Tsang Application and Theory 200 and sympy calculations.

        Coefficients within the phase function are::

            M  = [P[v, v]  P[v, h] -P[v, u]]
                 [P[h, v]  P[h, h] -P[h, u]]
                 [P[u, v]  P[u, h] P[u, u]]

        :param m: mode for decomposed phase matrix (0, 1, 2)
        :param mu: vector of cosines of incidence angle
        :returns: thephase matrix

        """

        assert mu_s is mu_i  # temporary hack, to be propagated
        mu = mu_i

        if npol is None:
            npol = 2 if m_max == 0 else 3

        P = smrt_matrix.empty((npol, npol, m_max + 1, len(mu_s), len(mu_i)))

        mu2 = mu**2

        v, h, u = 0, 1, 2

        raise SMRTError("There is a sign error in Tsang's book.")

        P[v, v, 0] = 0.5 * np.outer(mu2, mu2) + np.outer(1 - mu2, 1 - mu2)
        P[v, h, 0] = 0.5 * mu2[:, np.newaxis]  # mu2[:, np.newaxis]  # equiv np.dot(mu2, np.ones_like(mu2.T))
        if npol >= 3:
            P[v, u, 0] = 0

        P[h, v, 0] = P[v, h, 0].T
        P[h, h, 0] = 0.5
        if npol >= 3:
            P[h, u, 0] = 0

        if npol >= 3:
            P[u, v, 0] = 0
            P[u, h, 0] = 0
            P[u, u, 0] = 0  # this one is not null !!! set to zero here for simplificity but is 2*outer(mu, mu)

        if m_max >= 1:
            sint = np.sqrt(1. - mu2)
            cossint = mu * sint

            P[v, v, 1] = 2 * np.outer(cossint, cossint)
            P[v, h, 1] = 0
            P[v, u, 1] = np.outer(cossint, sint)

            P[h, v, 1] = 0
            P[h, h, 1] = 0
            P[h, u, 1] = 0

            P[u, v, 1] = 2 * P[v, u, 1].T  # does not work
            #P[u, v, 1] = -P[u, v, 1]      # This line is needed, I don't understand why!!!!!!!!!!!!!!!!!!!

            P[u, h, 1] = 0
            P[u, u, 1] = np.outer(sint, sint)

        if m_max >= 2:
            P[v, v, 2] = 0.5 * np.outer(mu2, mu2)
            P[v, h, 2] = -0.5 * mu2[:, np.newaxis]
            P[v, u, 2] = 0.5 * np.outer(mu2, mu)

            P[h, v, 2] = P[v, h, 2].T
            P[h, h, 2] = 0.5
            P[h, u, 2] = 0.5 * mu[np.newaxis, :]

            P[u, v, 2] = 2 * P[v, u, 2].T
            #P[u, v, 2] = -P[u, v, 2]      # This line is need, I don't understand why!!!!!!!!!!!!!!!!!!!

            P[u, h, 2] = mu[:, np.newaxis]
            #P[u, h, 2] = -P[u, h, 2]      # This line is need, I don't understand why!!!!!!!!!!!!!!!!!!!
            P[u, u, 2] = 0

        if m_max >= 3:
            P[:, :, 3:, :, :] = 0

        if npol == 3:
            P[v, u, :] = -P[v, u, :]  # minus comes from even phase function
            P[h, u, :] = -P[h, u, :]  # minus comes from even phase function

        # this normalisation is compatible with the 1/4pi normalisation used for the RT equation.
        coef = 3 * self.ks / 2   # no*fo^2 / Ks (see TsangI 3.2.49)

        return P * coef

    # the formulation taken by Ulaby seems to be the only one to be error-free !
    # we use this one
    ft_even_phase = ft_even_phase_baseonUlaby

    def phase(self, mu_s, mu_i, dphi, npol=2):

        p, sin_half_scatt = rayleigh_scattering_matrix_and_angle(mu_s, mu_i, dphi, npol)

        return smrt_matrix(1.5 * self.ks * p)

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

        return extinction_matrix(self.ks + self.ka, mu=mu, npol=npol)

    def effective_permittivity(self):
        return self._effective_permittivity
