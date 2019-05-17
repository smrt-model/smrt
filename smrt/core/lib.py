

import numpy as np
import scipy.sparse


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
