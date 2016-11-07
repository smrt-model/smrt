# coding: utf-8

"""Exponential autocorrelation function model of the microstructure. This microstructure model is used by MEMLS when IBA is selected.

parameters: frac_volume, corr_length

"""

import numpy as np

# local import
from ..core.globalconstants import DENSITY_OF_ICE
from ..core.error import SMRTError
from .autocorrelation import Autocorrelation

class Exponential(Autocorrelation):

    # TODO. Make 3D
    # TODO. Think about density - currently required
    # TODO. SSA as an alternative input or calculated
    # TODO. Make tests

    args = ["frac_volume", "corr_length"]
    optional_args = {}

    def __init__(self, params):

        super(Exponential, self).__init__(params)  # don't forget this line in our classes!

        # value of the correlation function at the origin
        self.corr_func_at_origin = self.frac_volume * (1.0 - self.frac_volume)

        # inverse slope of the normalized correlation function at the origin
        self.inv_slope_at_origin = self.corr_length

    def basic_check(self):
        """check consistency between the parameters"""
        pass

    def compute_ssa(self):
        """compute the ssa for the exponential model according to Debye 1957. See also Maetzler 2002 Eq. 11"""
        return 3 * (1-self.frac_volume) / (DENSITY_OF_ICE * self.corr_length)

    def autocorrelation_function(self, r):
        """compute the real space autocorrelation function"""
        f_real = self.corr_func_at_origin * np.exp(-r / self.corr_length)
        return f_real

    def ft_autocorrelation_function(self, k):
        """compute the fourier transform of the autocorrelation function analytically"""
        ft = self.corr_func_at_origin * 8 * np.pi * self.corr_length**3 / (1. + (k * self.corr_length)**2)**2
        return ft
