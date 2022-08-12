# coding: utf-8

"""Homogeneous microstructure. This microstructure model is to be used with non-scattering emmodel.

parameters: none

"""

# local import
from .autocorrelation import Autocorrelation


class Homogeneous(Autocorrelation):

    args = ['frac_volume']
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
        return 0

    def basic_check(self):
        """check consistency between the parameters"""
        pass

    def compute_ssa(self):
        """compute the ssa of an homogeneous medium"""
        return 0

    def autocorrelation_function(self, r):
        """compute the real space autocorrelation function"""
        f_real = 0
        return f_real

    def ft_autocorrelation_function(self, k):
        """compute the fourier transform of the autocorrelation function analytically"""
        ft = 0
        return ft
