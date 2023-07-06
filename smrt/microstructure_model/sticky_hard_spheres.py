# coding: utf-8

"""Monodisperse sticky hard sphere model of the microstructure.

parameters: frac_volume, radius, stickiness.

The stickiness is optional but it is recommended to use value around 0.2 as a first guess.
Be aware that low values of stickiness are invalid, the limit depends on the fractional volume
(see for instance Loewe and Picard, 2015). See the :py:meth:`~StickyHardSpheres.tau_min` method.

Currently the implementation is specific to ice / snow. It can not be used for other materials.
"""

import numpy as np

from ..core.globalconstants import DENSITY_OF_ICE
from ..core.error import SMRTError

from .autocorrelation import Autocorrelation


class StickyHardSpheres(Autocorrelation):
    """
    """
    args = ["frac_volume", "radius"]
    optional_args = {"stickiness": 1000}

    def __init__(self, params):

        super().__init__(params)  # don't forget this line in our classes!
        # self.basic_check()

    @property
    def corr_func_at_origin(self):
        # value of the correlation function at the origin
        return self.frac_volume * (1.0 - self.frac_volume)

    @property
    def inv_slope_at_origin(self):
        # inverse slope of the normalized correlation function at the origin
        return 4.0 / 3 * self.radius * (1 - self.frac_volume)

    def basic_check(self):
        """check consistency between the parameters"""
        if (self.stickiness < self.tau_min(self.frac_volume)):
            raise SMRTError("For volume fraction " + str(self.frac_volume)
                            + " the stickiness must be greater than "
                            + str(self.tau_min(self.frac_volume)))

    def compute_ssa(self):
        """Compute the ssa of a sphere assembly"""
        return 3.0 / (DENSITY_OF_ICE * self.radius)

    # No analytical function exists
    # def autocorrelation_function(self, r):
    #

    def ft_autocorrelation_function(self, k):
        """Compute the 3D Fourier transform of the isotropic correlation
        function for sticky hard spheres in Percus--Yevick
        approximation for given magnitude k of the 3D wave vector
        (float).

        """

        # TODO LH:
        # * get solution for t directly from method compute_t
        # (this would include a check if the combination of stickiness
        # and volume fraction is admissible.
        # * check if k is positive (maybe not required since the function is even in k)

        d = 2 * self.radius
        phi_2 = self.frac_volume
        tau = self.stickiness

        # scaling variable, Eq 32, LP2015
        X = np.atleast_1d(k) * d / 2.0

        # solution of the quadratic equation, Eq. 32, LP2015
        if np.isfinite(tau) and phi_2 > 0.:
            t = ((6 * tau * phi_2 - 6 * phi_2 - 6 * tau + (36 * tau**2 * phi_2**2 - 72 * tau * phi_2**2
                                                           - 72 * tau**2 * phi_2 + 30 * phi_2**2
                                                           + 72 * tau * phi_2 + 36 * tau**2 - 12 * phi_2)**0.5) / (phi_2 * (-1 + phi_2)))
        else:
            t = 0
        # sphere volume
        vd = 4.0 / 3 * np.pi * (d / 2.0)**3

        # intersection volume, Eq. 27, LP2015
        sqrt_vint__vd = np.empty_like(X)
        zerok = np.isclose(X, 0, atol=1e-03)
        nzerok = ~zerok
        # sqrt(intersection volume ) / X² per vd
        sqrt_vint__vd[nzerok] = 3 * (np.sinc(X[nzerok] / np.pi) - np.cos(X[nzerok])) / X[nzerok]**2
        sqrt_vint__vd[zerok] = 1

        # Ghislain says: the following quantities are already multiplied by the sqrt_vint_X2 to avoid singularity in 0.
        # this differs from the original equations where the vint is multiplied at the end.

        # auxiliary quantities defined in Tsang II Eq.8.4.19-8.4.22
        Psi_tsang_vol_vd = np.sinc(X / np.pi) / sqrt_vint__vd
        Phi_tsang_vol_vd = 1.0  # Ghislain says: after simple math, vint sqrt_vint_X2 simplifies

        # auxiliary quantities Eq 31, LP2015
        A_tsang_vol_vd = phi_2 / (1 - phi_2) * (
            (1 - t * phi_2 + 3 * phi_2 / (1 - phi_2)) * Phi_tsang_vol_vd
            + (3 - t * (1 - phi_2)) * Psi_tsang_vol_vd) + np.cos(X) / sqrt_vint__vd

        B_tsang_vol_vd = phi_2 / (1 - phi_2) * X * Phi_tsang_vol_vd + np.sin(X) / sqrt_vint__vd

        # structure factor Eq 31, LP2015
        S_tsang_vol__vd2 = 1 / (A_tsang_vol_vd**2 + B_tsang_vol_vd**2)

        # FT correlation function, Eq. 25, LP2015
        Ctilde = phi_2 * vd * S_tsang_vol__vd2

        # set limit value at k=0 manually, Eq. 33, LP2015
        # zerok = np.isclose(X, 0)
        # Ctilde[zerok] = (n * vd**2 / (phi_2 / (1-phi_2) * ((1 - t*phi_2 + 3 * phi_2 / (1 - phi_2)) + (3 - t * (1 - phi_2))) + 1)**2)
        Ctilde[zerok] = phi_2 * vd / (phi_2 / (1 - phi_2) * ((1 - t * phi_2 + 3 * phi_2 / (1 - phi_2)) + (3 - t * (1 - phi_2))) + 1)**2

        return Ctilde

    def compute_t(self):
        """compute the t parameter used in the stickiness"""

        if self.stickiness == np.inf:  # none-sticky case
            return 0

        f = self.frac_volume

        # calculate t

        # solve equation 8.4.22  Tsang vol II

        a = f / 12.0
        b = -(self.stickiness + f / (1 - f))
        c = (1 + f / 2) / (1 - f)**2

        discr2 = b**2 - 4 * a * c
        if np.any(discr2 < 0):
            raise SMRTError("negative discriminant")

        discr = np.sqrt(discr2)

        t = (- b - discr) / (2 * a)

        # check mhu<1+2f
        mhu = t * f * (1 - f)
        mhulim = 1 + 2 * f

        if mhu > mhulim:
            t = (-b + discr) / (2 * a)
            mhu = t * f * (1 - f)

        if mhu > mhulim:
            print(mhu, mhulim)
            raise SMRTError("no solution for the t parameter. Revise the stickiness")

        return t

    def tau_min(self, frac_volume):
        """compute the minimum possible stickiness value for given ice volume
        fraction

        """
        return (1.0 / 12
                * (14.0 * frac_volume**2 - 4 * frac_volume - 1)
                / (2 * frac_volume**2 - frac_volume - 1))
