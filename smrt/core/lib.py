

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


class smrt_diag(object):
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

    def __rmul__(self, other):
        return other * self.diag

    def __mul__(self, other):
        return self.diag * other

    def __getitem__(self, key):
        try:
            i, j = key
        except TypeError:
            raise IndexError("The index of a diag object must be a tuple with two indices. See smrt.core.lib for the rational of this diag object.")
        return self.diag[i] if i == j else 0

    def check_type(self, other):
        if not isinstance(other, np.ndarray) or other.ndim != 2:
            raise Runtime("multiplication with diag is only implemented for 2-d ndarray")


class smrt_matrix(object):
    """SMRT uses two formats of matrix: one most suitable to implement emmodel where equations are different for each polarization and another one suitable
    for DORT computation where stream and polarization are collapsed in one dimension to allow matrix operation. In addition, the reflection and transmission matrix
    are often diagonal matrix, which needs to be handled because it saves space and allow much faster operations. This class implemented all these features.

    """

    def __init__(self, mat, mtype=None):

        if mat is 0:
            self.mat = 0
            self.mtype = "0"
        else:
            self.mat = mat

            if mtype is None:
                if isinstance(mat, list) and len(mat) in [2, 3]:
                    # diagonal matrix
                    if len(mat[0].shape) == 2:
                        mtype = "diagonal5"
                    else:
                        mtype = "diagonal4"
                elif len(mat.shape) == 5:
                    mtype = "dense5"
                elif len(mat.shape) == 4:
                    mtype = "dense4"
                elif len(mat.shape) == 3:
                    mtype = "diagonal5"
                elif len(mat.shape) == 2:
                    mtype = "diagonal4"
                else:
                    raise SMRTError("Unsupported matrix size")
            self.mtype = mtype

    @staticmethod
    def empty(dims, mtype=None):
        mat = np.empty(dims)
        return smrt_matrix(mat, mtype)

    @staticmethod
    def zeros(dims, mtype=None):
        mat = np.zeros(dims)
        return smrt_matrix(mat, mtype)

    @staticmethod
    def ones(dims, mtype=None):
        mat = np.ones(dims)
        return smrt_matrix(mat, mtype)

    def compress(self, mode=None, auto_reduce_npol=False):
        """compress a matrix. This comprises several actions:
        1) select one mode, if relevant (dense5, and diagonal5).
        2) reduce the number of polarization from 3 to 2 if mode==0 and auto_reduce_npol=True.
        3) convert the format of the matrix to compressed numpy, involving a change of the dimension order (pola and streams are merged).

"""
        if self.mat is 0:
            return 0

        if self.mtype == "dense5":
            if mode is not None:
                return self.sel(mode=mode, auto_reduce_npol=auto_reduce_npol).compress()

            else:
                raise NotImplementedError
                # reorder from pola_s, pola_i, m, mu_s, mu_i to  m, mu_s, pola_s, mu_i, pola_i
                #mat = np.moveaxis(self.mat, (0, 1), (2, 4)) # 0 becomes 2, 1 becomes 4
                # merge mu_s * pola_s and mu_i * pola_i
                #return smrt_matrix(np.reshape(mat, (mat.shape[0], mat.shape[1]*mat.shape[2], mat.shape[3]*mat.shape[4])), mtype="compressed3")
    
        elif self.mtype == "dense4":
            if self.mat.shape[0] == 3 and auto_reduce_npol and mode == 0:
                ## 3pol->2pol
                mat = self.mat[0:2, 0:2, :, :]
            else:
                mat = self.mat

            # reorder from pola_s, pola_i, mu_s, mu_i to  mu_s, pola_s, mu_i, pola_i
            assert(len(mat.shape) == 4)
            mat = np.moveaxis(mat, (0, 1), (1, 3)) # 0 becomes 1, 1 becomes 3, so 2 becomes 0 and 3 becomes 2
            # merge mu_s * pola_s and mu_i * pola_i
            return np.reshape(mat, (mat.shape[0]*mat.shape[1], mat.shape[2]*mat.shape[3])) # return an 2x2 array !

        elif self.mtype == "diagonal5":
            if mode is not None:
                return self.sel(mode=mode, auto_reduce_npol=auto_reduce_npol).compress()
            else:
                raise NotImplementedError

        elif self.mtype == "diagonal4":
            if self.mat.shape[0] == 3 and auto_reduce_npol and mode == 0:
                ## 3pol->2pol
                mat = self.mat[0:2, :]
            else:
                mat = self.mat
            # reorder from pola, mu to mu*pola and compress
            assert(len(mat.shape) == 2)
            return smrt_diag(np.reshape(np.transpose(mat), mat.shape[0] * mat.shape[1])).as_dia_matrix()

        else:
            raise NotImplementedError
        # if m_max == 0:
        #     # active # this is a bit tricky because for m we need to go back to npol=2. This is probably unnecessary complex...
        #     self.ft_even_phase = dict()
        #     for m in range(m_max + 1):
        #         pp = p[0:2, 0:2, m] if m == 0 else p[:, :, m]
        #         pp = np.moveaxis(pp, (0, 1), (1, 3)) # 0 becomes 1, 1 becomes 3
        #         self.ft_even_phase[m] = np.reshape(pp, (pp.shape[0]*pp.shape[1], pp.shape[2]*pp.shape[3]))

    def __rmul__(self, other):
        return smrt_matrix(other * self.mat)

    def __mul__(self, other):
        return smrt_matrix(self.mat * other)

    def __truediv__(self, other):
        return smrt_matrix(self.mat / other)

    def __add__(self, other):
        if isinstance(other, smrt_matrix):
            return smrt_matrix(other.mat + self.mat)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, smrt_matrix):
            return smrt_matrix(other.mat - self.mat)
        else:
            raise NotImplementedError

    def __abs__(self):
        return np.abs(self.mat)

    def __getitem__(self, key):
        return self.mat[key]

    def __setitem__(self, key, v):
        self.mat[key] = v

    def sel(self, **kwargs):

        if 'mode' in kwargs:
            if self.mtype == "dense5":

                if self.mat.shape[0] == 3 and kwargs['auto_reduce_npol'] and kwargs['mode'] == 0:
                    ## 3pol->2pol
                    return smrt_matrix(self.mat[0:2, 0:2, kwargs['mode'], :, :], mtype='dense4')
                else:
                    return smrt_matrix(self.mat[:, :, kwargs['mode'], :, :], mtype='dense4')

            elif self.mtype == "diagonal5":
                if self.mat.shape[0] == 3 and kwargs['auto_reduce_npol'] and kwargs['mode'] == 0:
                    ## 3pol->2pol
                    return smrt_matrix(self.mat[0:2, kwargs['mode'], :], mtype='diagonal4')
                else:
                    return smrt_matrix(self.mat[:, kwargs['mode'], :], mtype='diagonal4')

            elif self.mtype == "dense4":
                raise SMRTError("Dense4 matrix can not be selected by mode")

            elif self.mtype == "diagonal4":
                raise SMRTError("Diagonal4 matrix can not be selected by mode")
            else:
                raise NotImplementedError 
        else:
            raise SMRTError("Currently only selection by mode is implemented")