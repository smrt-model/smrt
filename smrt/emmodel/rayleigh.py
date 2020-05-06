# coding: utf-8

""" Compute Rayleigh scattering. This theory requires the scatterers to be smaller than the wavelength and 
the medium to be sparsely populated (eq. very low density in the case of snow).

This model is only compatible with the Independent Sphere microstructure model

"""

import numpy as np

from ..core.error import SMRTError
from ..core.globalconstants import C_SPEED
from ..core.lib import smrt_matrix


class Rayleigh(object):
    """
    """
    def __init__(self, sensor, layer):

        # check here the limit of the Rayleigh model

        f = layer.frac_volume

        e0 = layer.permittivity(0, sensor.frequency)  # background permittivity
        eps = layer.permittivity(1, sensor.frequency)  # scatterer permittivity

        self._effective_permittivity = e0  # Rayleigh is for sparse medium only

        if abs(e0) > abs(eps):
            raise SMRTError("Rayleigh model (as implemented) is unable to handle background permittivity higher than scatterers permittivity. This error is probably due to dense snow layers. Limit the density of snow layer if possible.")

        # TODO Ghi: solve the problem of dielectric constant dependency. Which object is responsible of running
        # Probably the emmodule

        lmda = C_SPEED / sensor.frequency

        radius = layer.microstructure.radius

        k0 = 2 * np.pi / lmda
        self.ks = f * 2 * abs((eps - e0) / (eps + 2 * e0))**2 * radius**3 * k0**4
        self.ka = f * 9 * k0 * eps.imag / e0 * abs(e0 / (eps + 2 * e0))**2

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

        assert mu_s is mu_i  # temporary hack, to be propagated
        mu = mu_i

        if npol is None:
            npol = 2 if m_max == 0 else 3

        P = smrt_matrix.empty((npol, npol, m_max + 1, len(mu_s), len(mu_i)))

        mu2 = mu**2

        v, h, u = 0, 1, 2

        # mode m == 0
        P[v, v, 0] = 0.5 * np.outer(mu2, mu2) + np.outer(1 - mu2, 1 - mu2)
        P[v, h, 0] = 0.5 * mu2[:, np.newaxis]  # mu2[:, np.newaxis]  # equiv np.dot(mu2, np.ones_like(mu2.T))
        if npol >= 3:
            P[v, u] = 0

        P[h, v, 0] = P[v, h, 0].T
        P[h, h, 0] = 0.5
        if npol >= 3:
            P[h, u, 0] = 0

        if npol >= 3:
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

            P[u, v, 1] = -2 * P[v, u, 1].T

            P[u, h, 1] = 0
            P[u, u, 1] = np.outer(sint, sint)

        if m_max >= 2:
            P[v, v, 2] = 0.5 * np.outer(mu2, mu2)
            P[v, h, 2] = -0.5 * mu2[:, np.newaxis]
            P[v, u, 2] = 0.5 * np.outer(mu2, mu)

            P[h, v, 2] = P[v, h, 2].T
            P[h, h, 2] = 0.5
            P[h, u, 2] = -0.5 * mu[np.newaxis, :]

            P[u, v, 2] = -2 * P[v, u, 2].T
            P[u, h, 2] = mu[:, np.newaxis]
            P[u, u, 2] = np.outer(mu, mu)

        if m_max >=3:
            P[:, :, 3:, :, :] = 0

        if npol == 3:
            P[v, u, :] = -P[v, u, :]  # minus comes from even phase function
            P[h, u, :] = -P[h, u, :]  # minus comes from even phase function

        # this normalisation is compatible with the 1/4pi normalisation used for the RT equation.
        coef = 3 * self.ks / 2   # no*fo^2 / Ks (see TsangI 3.2.49)

        return P * coef

    def ft_even_phase_basedonJin(self, mu_s, mu_i, m_max, npol=None):
        """Rayleigh phase matrix.

        These are the Fourier decomposed phase matrices for modes m = 0, 1, 2.
        It is based on Y.Q. Jin

        Coefficients within the phase function are:
        ::

        M  = [Pvvp  Pvhp]
             [Phvp  Phhp]

        Inputs are:
        :param m: mode for decomposed phase matrix (0, 1, 2)
        :param mu: vector of cosines of incidence angle

        Returns P: phase matrix

        """
        assert mu_s is mu_i  # temporary hack, to be propagated
        mu = mu_i

        if npol is None:
            npol = 2 if m == 0 else 3

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
        """Rayleigh phase matrix.

        These are the Fourier decomposed phase matrices for modes m = 0, 1, 2.
        Equations are from p128 Tsang Application and Theory 200 and sympy calculations

        Coefficients within the phase function are:
        ::

        M  = [P[v, v]  P[v, h] -P[v, u]]
             [P[h, v]  P[h, h] -P[h, u]]
             [P[u, v]  P[u, h] P[u, u]]

        Inputs are:
        :param m: mode for decomposed phase matrix (0, 1, 2)
        :param mu: vector of cosines of incidence angle

        Returns P: phase matrix

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
        # Tsang theory and application p271 Eq 7.2.16

        mu_s = np.atleast_1d(mu_s)[np.newaxis, :, np.newaxis]
        mu_i = np.atleast_1d(mu_i)[np.newaxis, np.newaxis, :]

        dphi = np.atleast_1d(dphi)
        sinphi = np.sin(dphi)[:, np.newaxis, np.newaxis]
        cosphi = np.cos(dphi)[:, np.newaxis, np.newaxis]

        # Tsang theory and application p127 Eq 3.2.47
        fvv = cosphi * mu_s * mu_i + np.sqrt(1 - mu_s**2) * np.sqrt(1 - mu_i**2)
        fhv = - sinphi * mu_i   # there is an error in Tsang book. By symmetry of the equation fvh has sin(phi_s-phi_i) and fhv has sin(phi_i-phi_s)
        fhh = cosphi
        fvh = sinphi * mu_s

        if npol == 2:
            p = [[fvv * fvv, fvh * fvh],
                 [fhv * fhv, fhh * fhh]]

        elif npol == 3:
            p = [[fvv * fvv, fvh * fvh, fvh * fvv],
                 [fhv * fhv, fhh * fhh, fhv * fhh],
                 [2 * fvv * fhv, 2 * fvh * fhh, fvv * fhh + fvh * fhv]]
        else:
            raise RuntimeError("invalid number of polarisation")

        # broadcast, it is not automatic (anymore?)
        shape = dphi.size, mu_s.size, mu_i.size
        p = [[np.broadcast_to(ppp, shape) for ppp in pp] for pp in p]

        return smrt_matrix(1.5 * self.ks * np.array(p))

    def ke(self, mu):
        """return the extinction coefficient"""
        return np.full(len(mu), self.ks + self.ka)

    def effective_permittivity(self):
        return self._effective_permittivity
