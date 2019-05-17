# coding: utf-8

""" Compute Rayleigh scattering. This theory requires the scatterers to be smaller than the wavelength and 
the medium to be sparsely populated (eq. very low density in the case of snow).

This model is only compatible with the Independent Sphere microstructure model

"""

import numpy as np

from ..core.error import SMRTError
from ..core.globalconstants import C_SPEED


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

    def ft_even_phase_baseonUlaby(self, m, mu_s, mu_i, npol=None):
        """#
        # Equations are from pg 1188-1189 Ulaby, Moore, Fung. Microwave Remote Sensing Vol III.
        # See also pg 157 of Tsang, Kong and Shin: Theory of Microwave Remote Sensing (1985) - can be used to derive
        # the Ulaby equations.

        """

        assert mu_s is mu_i  # temporary hack, to be propagated
        mu = mu_i

        if npol is None:
            npol = 2 if m == 0 else 3

        mu2 = mu**2

        if m == 0:
            PCvvp = 0.5 * np.outer(mu2, mu2) + np.outer(1 - mu2, 1 - mu2)
            PCvhp = 0.5 * mu2[:, np.newaxis]  #mu2[:, np.newaxis]  # equiv np.dot(mu2, np.ones_like(mu2.T))
            PSvup = 0

            PChvp = PCvhp.T
            PChhp = 0.5
            PShup = 0

            PSuvp = 0
            PSuhp = 0
            PCuup = 0

        elif m == 1:
            sint = np.sqrt(1. - mu2)
            cossint = mu * sint

            PCvvp = 2*np.outer(cossint, cossint)
            PCvhp = 0
            PSvup = np.outer(cossint, sint)

            PChvp = 0
            PChhp = 0
            PShup = 0

            PSuvp = -2*PSvup.T

            PSuhp = 0
            PCuup = np.outer(sint, sint)

        elif m == 2:
            PCvvp = 0.5*np.outer(mu2, mu2)
            PCvhp = -0.5 * mu2[:, np.newaxis]
            PSvup = 0.5 * np.outer(mu2, mu)

            PChvp = PCvhp.T
            PChhp = 0.5
            PShup = -0.5 * mu[np.newaxis, :]

            PSuvp = -2*PSvup.T
            PSuhp = mu[:, np.newaxis]
            PCuup = np.outer(mu, mu)

        else:
            return 0 #raise Exception("Rayleigh mode should be equal to 0, 1 or 2")

        # this normalisation is compatible with the 1/4pi normalisation used for the RT equation.
        coef = 3 * self.ks / 2   # no*fo^2 / Ks (see TsangI 3.2.49)

        n = len(mu)
        P = np.empty((npol * n, npol * n))
        P[0::npol, 0::npol] = PCvvp * coef
        P[0::npol, 1::npol] = PCvhp * coef

        P[1::npol, 0::npol] = PChvp * coef
        P[1::npol, 1::npol] = PChhp * coef

        if npol == 3:
            P[0::npol, 2::npol] = -PSvup * coef  # minus comes from even phase function
            P[1::npol, 2::npol] = -PShup * coef  # minus comes from even phase function
            P[2::npol, 0::npol] = PSuvp * coef
            P[2::npol, 1::npol] = PSuhp * coef
            P[2::npol, 2::npol] = PCuup * coef

        return P

    def ft_even_phase_basedonJin(self, m, mu_s, mu_i, npol=None):
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


        mu2 = mu**2

        if m == 0:
            PCvvp = 0.5 * np.outer(mu2, mu2) + np.outer(1 - mu2, 1 - mu2)
            PCvhp = 0.5 * mu2[:, np.newaxis]  # equiv np.dot(mu2, np.ones_like(mu2.T))
            PSvup = 0

            PChvp = PCvhp.T
            PChhp = 0.5
            PShup = 0

            PSuvp = 0
            PSuhp = 0
            PCuup = 0

        elif m == 1:
            sint = np.sqrt(1. - mu2)
            cossint = mu * sint

            PCvvp = 2*np.outer(cossint, cossint)
            PCvhp = 0
            PSvup = np.outer(cossint, sint)

            PChvp = 0
            PChhp = 0
            PShup = 0

            PSuvp = -2*PSvup.T  # could transpose Pvu
            PSuhp = 0
            PCuup = np.outer(sint, sint)

        elif m == 2:
            PCvvp = 0.5*np.outer(mu2, mu2)
            PCvhp = -0.5 * mu2[:, np.newaxis]  # equiv - np.dot(mu2, np.ones_like(mu2.T))
            PSvup = 0.5 * np.outer(mu2, mu)

            PChvp = PCvhp.T
            PChhp = 0.5
            PShup = -0.5 * mu[np.newaxis, :]  ## error of theta_i theta_s in Y.Q. Jin

            PSuvp = - np.outer(mu, mu2)
            PSuhp = mu[:, np.newaxis]
            PCuup = -np.outer(mu, mu)
            PCuup = 0   # error in Y.Q. Jin ?? According to Tsang this term is null

        else:
            return 0  #           raise Exception("Rayleigh mode should be equal to 0, 1 or 2")

        # this normalisation is compatible with the 1/4pi normalisation used for the RT equation.
        coef = 3 * self.ks / 2

        n = len(mu)
        P = np.empty((npol * n, npol * n))
        P[0::npol, 0::npol] = PCvvp * coef
        P[0::npol, 1::npol] = PCvhp * coef

        P[1::npol, 0::npol] = PChvp * coef
        P[1::npol, 1::npol] = PChhp * coef

        if npol == 3:
            P[0::npol, 2::npol] = - PSvup * coef
            P[1::npol, 2::npol] = - PShup * coef

            P[2::npol, 0::npol] = PSuvp * coef
            P[2::npol, 1::npol] = PSuhp * coef
            P[2::npol, 2::npol] = PCuup * coef

        return P


    def ft_even_phase_tsang(self, m, mu_s, mu_i, npol=None):
        """Rayleigh phase matrix.

        These are the Fourier decomposed phase matrices for modes m = 0, 1, 2.
        Equations are from p128 Tsang Application and Theory 200 and sympy calculations

        Coefficients within the phase function are:
        ::

        M  = [PCvvp  PCvhp -PSvup]
             [PChvp  PChhp -PShup]
             [PSuvp  PSuhp PCuup]

        Inputs are:
        :param m: mode for decomposed phase matrix (0, 1, 2)
        :param mu: vector of cosines of incidence angle

        Returns P: phase matrix

        """

        assert mu_s is mu_i  # temporary hack, to be propagated
        mu = mu_i

        if npol is None:
            npol = 2 if m == 0 else 3

        mu2 = mu**2

        if m == 0:
            PCvvp = 0.5 * np.outer(mu2, mu2) + np.outer(1 - mu2, 1 - mu2)
            PCvhp = 0.5 * mu2[:, np.newaxis]  #mu2[:, np.newaxis]  # equiv np.dot(mu2, np.ones_like(mu2.T))
            PSvup = 0

            PChvp = PCvhp.T
            PChhp = 0.5
            PShup = 0

            PSuvp = 0
            PSuhp = 0
            PCuup = 0  # this one is not null !!! set to zero here for simplificity but is 2*outer(mu, mu)

        elif m == 1:
            sint = np.sqrt(1. - mu2)
            cossint = mu * sint

            PCvvp = 2*np.outer(cossint, cossint)
            PCvhp = 0
            PSvup = np.outer(cossint, sint)

            PChvp = 0
            PChhp = 0
            PShup = 0

            PSuvp = 2*PSvup.T  # does not work
            PSuvp = -PSuvp      # This line is needed, I don't understand why!!!!!!!!!!!!!!!!!!!

            PSuhp = 0
            PCuup = np.outer(sint, sint)

        elif m == 2:
            PCvvp = 0.5*np.outer(mu2, mu2)
            PCvhp = -0.5 * mu2[:, np.newaxis]
            PSvup = 0.5 * np.outer(mu2, mu)

            PChvp = PCvhp.T
            PChhp = 0.5
            PShup = 0.5 * mu[np.newaxis, :]

            PSuvp = 2*PSvup.T
            PSuvp = -PSuvp      # This line is need, I don't understand why!!!!!!!!!!!!!!!!!!!

            PSuhp = mu[:, np.newaxis]
            PSuhp = -PSuhp      # This line is need, I don't understand why!!!!!!!!!!!!!!!!!!!
            PCuup = 0

        else:
            return 0  # raise Exception("Rayleigh mode should be equal to 0, 1 or 2")

        # this normalisation is compatible with the 1/4pi normalisation used for the RT equation.
        coef = 3 * self.ks / 2   # no*fo^2 / Ks (see TsangI 3.2.49)

        n = len(mu)
        P = np.empty((npol * n, npol * n))
        P[0::npol, 0::npol] = PCvvp * coef
        P[0::npol, 1::npol] = PCvhp * coef

        P[1::npol, 0::npol] = PChvp * coef
        P[1::npol, 1::npol] = PChhp * coef

        if npol == 3:
            P[0::npol, 2::npol] = -PSvup * coef  # minus comes from even phase function
            P[1::npol, 2::npol] = -PShup * coef  # minus comes from even phase function
            P[2::npol, 0::npol] = PSuvp * coef
            P[2::npol, 1::npol] = PSuhp * coef
            P[2::npol, 2::npol] = PCuup * coef

        return P

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

        return 1.5 * self.ks * np.array(p).squeeze()

    def ke(self, mu):
        """return the extinction coefficient"""
        return np.full(len(mu), self.ks + self.ka)

    def effective_permittivity(self):
        return self._effective_permittivity
