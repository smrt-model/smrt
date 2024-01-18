# coding: utf-8

"""Gaussian Random field model of the microstructure.

parameters: frac_volume, corr_length, repeat_distance

"""

import numpy as np
from scipy.special import erfinv
#from scipy.special import erfcinv

from .autocorrelation import Autocorrelation


class GaussianRandomField(Autocorrelation):
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

    @property
    def inv_slope_at_origin(self):
        # inverse slope of the normalized correlation function at the origin
        beta = np.sqrt(2) * erfinv(2 * (1 - self.frac_volume) - 1)
        # second derivative of the field acf at the origin
        acf_psi_doubleprime = -1.0 / 2 * ((1.0 / self.corr_length)**2 + 1.0 / 3 * (2 * np.pi / self.repeat_distance)**2)
        SSA_tilde = 2.0 / np.pi * np.exp(- beta**2 / 2) * np.sqrt(-acf_psi_doubleprime) / self.frac_volume
        return 4.0 * (1 - self.frac_volume) / SSA_tilde

    def basic_check(self):
        """check consistency between the parameters"""
        pass

    def compute_ssa(self):
        """Compute the ssa for a sphere"""
        # implement here the calculation of the SSA from the microstructure parameters.
        raise Exception("to be implemented")

    def autocorrelation_function(self, r):
        """compute the real space autocorrelation function for the Gaussian random field model

        """
        # compute the cut-level parameter beta
        beta = np.sqrt(2) * erfinv(2 * (1-self.frac_volume) - 1)

        # the covariance of the GRF
        acf_psi = (np.exp(-r/self.corr_length) * (1 + r / self.corr_length)
                   * np.sinc(2*r/self.repeat_distance))

        # integral discretization. henning says: the resolution 1e-2 is ad hoc, test required,
        # the integrand has a (integrable) singularity for t=1 and acf_psi = 1, so an adaptive
        # discretization seems preferable -> TODO
        dt = 1e-2
        t = np.arange(0, 1, dt)

        # the gridded integrand, via change of integration variable
        # compared to the wp-2 docu, to enable array-based computation
        t_gridded, acf_psi_gridded = np.meshgrid(t, acf_psi)
        integrand_gridded = (acf_psi_gridded / np.sqrt(1 - (t_gridded * acf_psi_gridded)**2)
                             * np.exp(- beta**2 / (1 + t_gridded * acf_psi_gridded)))

        acf = 1.0 / (2 * np.pi) * np.trapz(integrand_gridded, x=t_gridded)

        return acf

    # ft not known analytically: deligate
    # def ft_autocorrelation_function(self, k):
