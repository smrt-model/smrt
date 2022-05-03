# coding: utf-8

"""Teubner Strey model model of the microstructure.

parameters: frac_volume, corr_length, repeat_distance

"""

import numpy as np

from .autocorrelation import Autocorrelation


class TeubnerStrey(Autocorrelation):
    """
    """
    args = ["frac_volume", "corr_length", "repeat_distance"]
    optional_args = {}

    def __init__(self, params):

        super().__init__(params)  # don't forget this line in our classes!

    @property
    def corr_func_at_origin(self):
        # value of the correlation function at the origin
        return self.frac_volume * (1.0 - self.frac_volume)

    def basic_check(self):
        """check consistency between the parameters"""
        pass

    def compute_ssa(self):
        """Compute the ssa for a sphere"""
        # implement here the calculation of the SSA from the microstructure parameters.
        raise NotImplementedError("to be implemented")

    def autocorrelation_function(self, r):
        """compute the real space autocorrelation function for the Teubner Strey model

        """

        acf = np.exp(-r / self.corr_length) * np.sinc(2 * r / self.repeat_distance)

        return self.corr_func_at_origin * acf

    def ft_autocorrelation_function(self, k):
        """Compute the 3D Fourier transform of the isotropic correlation
        function for Teubner Strey for given magnitude k of the 3D wave vector
        (float).

        """

        X = (k * self.corr_length)**2
        Y = (2 * np.pi * self.corr_length / self.repeat_distance)**2
        ft_acf_normalized = 8 * np.pi * self.corr_length**3 / ((1 + Y)**2 + 2 * (1 - Y) * X + X**2)

        return self.corr_func_at_origin * ft_acf_normalized
