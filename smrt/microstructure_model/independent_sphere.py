# coding: utf-8

"""Independent sphere model of the microstructure.

parameters: frac_volume, radius

"""

import numpy as np

from ..core.globalconstants import DENSITY_OF_ICE

from .autocorrelation import Autocorrelation


class IndependentSphere(Autocorrelation):

    args = ["frac_volume", "radius"]
    optional_args = {}

    def __init__(self, params):

        super().__init__(params)  # don't forget this line in our classes!

    @property
    def corr_func_at_origin(self):
        # value of the correlation function at the origin
        return self.frac_volume * (1.0 - self.frac_volume)

    @property
    def inv_slope_at_origin(self):
        # inverse slope of the normalized correlation function at the origin
        return 4.0 / 3 * self.radius

    def basic_check(self):
        """check consistency between the parameters"""
        pass

    def compute_ssa(self):
        """Compute the ssa for a sphere"""
        # implement here the calculation of the SSA from the microstructure parameters.
        return 3.0 / (DENSITY_OF_ICE * self.radius)

    def autocorrelation_function(self, r):
        """compute the real space autocorrelation function for an independent sphere

        """
        # spherical correlation function
        acf = np.empty_like(r)

        # the Heaviside factor
        heaviside = r <= 2 * self.radius

        acf[heaviside] = (1 - r[heaviside] / ((4 * self.radius) / 3) + r[heaviside]**3 / ((2*self.radius)**3 * 2))

        return self.corr_func_at_origin * acf

    def ft_autocorrelation_function(self, k):
        """Compute the 3D Fourier transform of the isotropic correlation
        function for an independent sphere for given magnitude k of the 3D wave vector
        (float).

        """

        X = self.radius * np.asarray(k)
        volume_sphere = 4.0 / 3 * np.pi * self.radius**3

        bessel_term = np.empty_like(X)
        zero_X = np.isclose(X, 0)
        non_zero_X = np.logical_not(zero_X)
        X_non_zero = X[non_zero_X]

        bessel_term[non_zero_X] = (9 * ((np.sin(X_non_zero) - X_non_zero * np.cos(X_non_zero))
                                        / X_non_zero**3)**2)
        bessel_term[zero_X] = 1.0
        return self.corr_func_at_origin * volume_sphere * bessel_term
