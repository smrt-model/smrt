

import collections
import six
import numpy as np
import pandas as pd
import scipy.sparse

from .error import SMRTError



def get(x, i, name=None):  # function to take the i-eme value in an array or dict of array. Can deal with scalar as well. In this case, it repeats the value.

    if isinstance(x, six.string_types):
        return x
    elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        if i >=len(x.values):
            raise SMRTError("The array '%s' is too short compared to the thickness array" % name)
        return x.values[i]
    if isinstance(x, collections.Sequence) or isinstance(x, np.ndarray):
        if i >=len(x):
            raise SMRTError("The array '%s' is too short compared to the thickness array.")
        return x[i]
    elif isinstance(x, dict):
        return {k: get(x[k], i, k) for k in x}
    else:
        return x


def is_sequence(x):
    # maybe not the smartest way...
    return (
            isinstance(x, collections.Sequence) or \
            isinstance(x, np.ndarray) or \
            isinstance(x, pd.DataFrame) or \
            isinstance(x, pd.Series)
            ) and not isinstance(x, six.string_types)


class diag(object):
    """Scipy.sparse is very slow for diagonal matrix and numpy has no good support for linear algebra. This diag class
    implements simple diagional object without numpy subclassing and without much features.
    It seems that proper subclassing numpy and overloading matmul is a very difficult problem."""

    __array_ufunc__ = None

    def __init__(self, arr):
        self.diag = arr

    # def sum(self):
    #     return self.diag.sum()

    def as_dia_matrix(self):
        return scipy.sparse.diags(self.diag, 0)

    def diagonal(self):
        return self.diag

    def __rmatmul__(self, other):
        self.check_type(other)
        return other * self.diag[np.newaxis, :]

    def __matmul__(self, other):
        self.check_type(other)
        return other * self.diag[:, np.newaxis]

    def __getitem__(self, key):
        try:
            i, j = key
        except TypeError:
            raise IndexError("The index of a diag object must be a tuple with two indices. See smrt.core.lib for the rational of this diag object.")
        return self.diag[i] if i == j else 0

    def check_type(self, other):
        if not isinstance(other, np.ndarray) or other.ndim != 2:
            raise Runtime("multiplication with diag is only implemented for 2-d ndarray")
