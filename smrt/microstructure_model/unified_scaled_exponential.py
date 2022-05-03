# coding: utf-8

"""Scaled exponential autocorrelation function model of the microstructure. This microstructure uses unified parameters as defined by 
G. Picard, H. Löwe, F. Domine, L. Arnaud, F. Larue, V. Favier, E. Le Meur, E. Lefebvre, J. Savarino, A. Royer, The snow microstructural control on microwave scattering, AGU Advances.

parameters: frac_volume, porod_length, polydispersity

"""

import numpy as np

# local import
from .unified_autocorrelation import UnifiedAutocorrelation


class UnifiedScaledExponential(UnifiedAutocorrelation):

    def __init__(self, params):

        super().__init__(params)  # don't forget this line in our classes!

        self.corr_length = self.polydispersity * self.porod_length

        # value of the correlation function at the origin
        self.corr_func_at_origin = self.frac_volume * (1.0 - self.frac_volume)

    def autocorrelation_function(self, r):
        """compute the real space autocorrelation function"""
        return self.corr_func_at_origin * np.exp(-r * self.corr_length)

    def ft_autocorrelation_function(self, k):
        """compute the fourier transform of the autocorrelation function analytically"""
        X = (k * self.corr_length)**2

        ft = self.corr_func_at_origin * 8 * np.pi * self.corr_length**3 / (1. + X)**2
        return ft
