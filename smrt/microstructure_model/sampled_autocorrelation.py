# coding: utf-8

"""Sampled autocorrelation function model. To use when no analytical form of the autocorrelation function but
the values of the autocorrelation function (`acf`) is known at a series of `lag`.

parameters: frac_volume, lag, acf

`acf` contains the values at different `lag`. These parameters must be lists or arrays.

"""

import numpy as np
import warnings

# local import
from .autocorrelation import Autocorrelation


class SampledAutocorrelation(Autocorrelation):

    # TODO. Make 3D
    # TODO. Think about density - currently required
    # TODO. SSA as an alternative input or calculated
    # TODO. Make tests

    args = ["frac_volume", "lag", "acf"]
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
        """compute the ssa according to Debye 1957. See also Maetzler 2002 Eq. 11"""
        pass

    def autocorrelation_function(self, r):
        """compute the real space autocorrelation function by interpolation of requested values from known values"""
        if(r[-1] > self.lag[-1]):
            warnings.warn("Warning: Autocorrelation function is computed out of range of known values")

        return np.interp(r, self.lag, self.acf)

    #def ft_autocorrelation_function(self, k):
