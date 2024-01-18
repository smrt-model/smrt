# coding: utf-8

"""
This module contains the base classes for the microstructure classes.
**It is not used directly**.
"""


import copy
import numpy as np


from scipy.fftpack import dst

from ..core.error import SMRTError


class AutocorrelationBase(object):

    """Low level base class for the Autocorrelation base class to handle optional and required arguments.
        **It should not be used directly**.

"""

    def __init__(self, params):

        super().__init__()

        if not hasattr(self, "all_optional_arguments"):
            self.compute_all_arguments()

        # register the parameters
        for arg in self.all_required_arguments:
            if arg in params:
                setattr(self, arg, params.get(arg))
            else:
                raise SMRTError("Parameter %s must be specified" % arg)

        for arg in self.all_optional_arguments:
            setattr(self, arg, params.get(arg, self.all_optional_arguments[arg]))

    @classmethod
    def compute_all_arguments(cls):
        # """return the list of valid arguments. this include those defined in the whole class hierarchy"""
        # TODO transfer this in a the __metaclass__ to avoid recomputation
        cls.all_required_arguments = []
        cls.all_optional_arguments = {}

        upcls = cls
        while upcls is not AutocorrelationBase:
            cls.all_required_arguments += getattr(upcls, "args", [])
            if hasattr(upcls, "optional_args"):
                cls.all_optional_arguments.update({k: v for k, v in upcls.optional_args.items() if k not in cls.all_optional_arguments})
            upcls = upcls.__base__   # may break if multiple inheritence... let us know if you have this problem

    @classmethod
    def valid_arguments(cls):
        if not hasattr(cls, "all_optional_arguments"):
            cls.compute_all_arguments()
        return cls.all_required_arguments + list(cls.all_optional_arguments.keys())


class Autocorrelation(AutocorrelationBase):

    """Base class for autocorrelation function classes. It should not be
used directly but sub-classed. It provides generic handling of the numerical fft and invfft when
required by the user or when necessary due to the lack of implementation of
the real or ft autocorrelation functions. See the source of :py:class:`~smrt.microstructure_model.exponential.Exponential`
to see how to use this class.

    """
    args = []
    optional_args = {'ft_numerical': False, 'real_numerical': False}

    def __init__(self, params):

        super().__init__(params)

        # numerical or not
        if not hasattr(self, "ft_autocorrelation_function") or params.get('ft_numerical', False):
            self.ft_autocorrelation_function = self.ft_autocorrelation_function_fft

        if not hasattr(self, "autocorrelation_function") or params.get('real_numerical', False):
            self.autocorrelation_function = self.autocorrelation_function_invfft

    def ft_autocorrelation_function_fft(self, k):
        """compute the fourier transform of the autocorrelation function via fft
        Args:
        k: array of wave vector magnitude values, ordered, and non-negative
        """

        k_abs = np.abs(k)

        #assert((np.diff(k) > 0).all())  # check k is sorted
        #assert((k > -np.finfo(float).eps).all())  # check k is non-negative

        # re-sampling
        # number of fourier auxiliary grid points, presently fixed
        N = 4096

        # grid resolution, fraction of the unique characteristic scale
        dr = self.inv_slope_at_origin / 20.0

        # compute correlation function and auxiliary wave vector arrray
        points = np.arange(N)
        r = dr * points
        C = self.autocorrelation_function(r)
        L = N * dr
        delta_k = np.pi / L
        k_resampled = delta_k * points

        # fft for auxiliary wave vector array
        ft_resampled = np.empty_like(C)
        ft_resampled[1:] = dst(4 * np.pi * C[1:] * r[1:], type=1) / (2 / dr * k_resampled[1:])
        ft_resampled[0] = dr * 4 * np.pi * np.sum(C * r**2)

        # get ft values for input k-values by linear interpolation
        ft = np.interp(k_abs, k_resampled, ft_resampled)
        return ft

    def autocorrelation_function_invfft(self, r):
        """Compute the autocorrelation function from an analytically known FT via fft
        Args:
        r: array of lag vector magnitude values, ordered, non-negative
        """

        assert((np.diff(r) > 0).all())  # check if r is sorted
        assert((r > -np.finfo(float).eps).all())  # check if r is non-negative

        # re-sampling
        if np.isclose(r[0], 0):
            r_spacing = r[1]         # alternative: np.min(diff(r))
        else:
            r_spacing = r[0]

        no_points = (np.max(r) - np.min(r)) / r_spacing
        points = np.arange(no_points)
        r_resampled = r_spacing * points

        dk = np.pi / (no_points * r_spacing)
        k = dk * points
        ft = self.ft_autocorrelation_function(k)

        C_resampled = np.empty_like(ft)
        C_resampled[1:] = dst(4 * np.pi * ft[1:] * k[1:], type=1) / (2 / (dk / (2 * np.pi)**3) * r_resampled[1:])
        C_resampled[0] = (dk / (2 * np.pi)**3) * 4 * np.pi * np.sum(ft * k**2)

        # get invft values corresponding to input r-values by linear interpolation
        C = np.interp(r, r_resampled, C_resampled)
        return C

    def inverted_medium(self):
        """return the same autocorrelation for the inverted medium. In general, it is only necessary to invert the fractional volume if
        the autocorrelation function is numerically symmetric as it should be. This needs to be reimplemented in the sub classes if this is 
        not sufficient.
        """

        obj = copy.copy(self)
        obj.frac_volume = 1.0 - self.frac_volume
        return obj
