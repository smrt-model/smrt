
import os

from collections.abc import Sequence
import numpy as np
import pandas as pd
import scipy.sparse
from smrt.core.optional_numba import numba

from smrt.core.error import SMRTError


def get(x, i, name=None):
    # function to take the i-eme value in an array or dict of array. Can deal with scalar as well. In this case, it repeats the value.

    if isinstance(x, str):
        return x
    elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        if i >= len(x.values):
            raise SMRTError("The array '%s' is too short compared to the thickness array" % name)
        return x.values[i]
    if isinstance(x, Sequence) or isinstance(x, np.ndarray):
        if i >= len(x):
            raise SMRTError("The array '%s' is too short compared to the thickness array." % name)
        return x[i]
    elif isinstance(x, dict):
        return {k: get(x[k], i, k) for k in x}
    else:
        return x


def check_argument_size(x, n, name=None):
    # this function check that x is either a scalar or a sequence of exactly n items

    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        error = len(x.values) != n
    elif (not isinstance(x, str) and isinstance(x, Sequence)) or isinstance(x, np.ndarray):
        error = len(x) != n
    elif isinstance(x, dict):
        for k in x:
            check_argument_size(x[k], n, k)
        return
    else:
        return
    if error:
        raise SMRTError("The array '%s' must be a scalar or have the same size as the 'thickness' array." % name)


def is_sequence(x):
    # maybe not the smartest way...
    return (
        isinstance(x, Sequence) or
        isinstance(x, np.ndarray) or
        isinstance(x, pd.DataFrame) or
        isinstance(x, pd.Series)
    ) and not isinstance(x, str)


def len_atleast_1d(x):
    try:
        return len(x)
    except TypeError:
        return 1 if x is not None else 0


class smrt_diag(object):
    """Scipy.sparse is very slow for diagonal matrix and numpy has no good support for linear algebra. This diag class
    implements simple diagonal object without numpy subclassing (but without much features).
    It seems that proper subclassing numpy and overloading matmul is a very difficult problem."""

    __array_ufunc__ = None

    def __init__(self, arr):
        self.diag = np.atleast_1d(arr)

        # self.shape = shape if shape is not None else (len(self.diag), len(self.diag))

    def diagonal(self):
        return self.diag

    @property
    def shape(self):
        return (len(self.diag), len(self.diag))

    def __len__(self):
        return len(self.diag)

    def __rmatmul__(self, other):
        self.check_type(other)
        # assert other.shape[1] == self.shape[0]
        return other * self.diag[np.newaxis, :]

    def __matmul__(self, other):
        # assert self.shape[1] == other.shape[0]
        if isinstance(other, smrt_diag):
            # return a diagonal object
            return smrt_diag(other.diag * self.diag)  # , shape=(self.shape[0], other.shape[1]))
        else:
            self.check_type(other)
            # other must be an ndarray
            return other * self.diag[:, np.newaxis]

    def __rmul__(self, other):
        assert np.isscalar(other)
        return smrt_diag(other * self.diag)

    def __mul__(self, other):
        assert np.isscalar(other)
        return smrt_diag(self.diag * other)

    def __add__(self, other):
        if isinstance(other, smrt_diag):
            return smrt_diag(other.diag + self.diag)
        elif np.isscalar(other) and other == 0:
            return self
        elif isinstance(other, np.ndarray):
            assert other.shape == self.shape  # we do not allow broadcasting (not yet)...
            other = other.copy()
            other[np.diag_indices_from[other]] += self.diag
            return other
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, smrt_diag):
            return smrt_diag(other.diag - self.diag)
        elif np.isscalar(other) and other == 0:
            return self
        else:
            raise NotImplementedError

    def __iadd__(self, other):
        if isinstance(other, smrt_diag):
            self.diag += other.diag
            return self
        else:
            raise NotImplementedError

    def __isub__(self, other):
        if isinstance(other, smrt_diag):
            self.diag -= other.diag
            return self
        else:

            raise NotImplementedError

    def __getitem__(self, key):
        try:
            i, j = key
        except TypeError:
            raise IndexError("The index of a diag object must be a tuple with two indices. See smrt.core.lib for the rational of this diag object.")
        return self.diag[i] if i == j else 0

    def check_type(self, other):
        if not isinstance(other, np.ndarray) or other.ndim != 2:
            raise NotImplementedError("multiplication with diag is only implemented for 2-d ndarray")


class smrt_matrix(object):
    """SMRT uses two formats of matrix: one most suitable to implement emmodel where equations are different for each
    polarization and another one suitable for DORT computation where stream and polarization are collapsed in one
    dimension to allow matrix operation. In addition, the reflection and transmission matrix are often diagonal matrix,
    which needs to be handled because it saves space and allow much faster operations. This class implemented all these
    features.

    """

    def __init__(self, mat, mtype=None):

        if is_zero_scalar(mat):
            self.values = np.float64(0.)  # 0, but can be used as a numpy thing
            self.mtype = "0"
        else:
            self.values = mat

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

    @staticmethod
    def full(dims, value, mtype=None):
        mat = np.full(dims, value)
        return smrt_matrix(mat, mtype)

    @property
    def npol(self):
        return self.values.shape[0]

    def is_equal_zero(self):
        return is_equal_zero(self)

    def compress(self, mode=None, auto_reduce_npol=False):
        """compress a matrix. This comprises several actions:
        1) select one mode, if relevant (dense5, and diagonal5).
        2) reduce the number of polarization from 3 to 2 if mode==0 and auto_reduce_npol=True.
        3) convert the format of the matrix to compressed numpy, involving a change of the dimension order (pola and streams are merged).

"""
        if self.mtype == "0":
            return np.float64(0.)  # 0, but can be used as a numpy thing

        if self.mtype == "dense5":
            if mode is not None:
                return self.sel(mode=mode, auto_reduce_npol=auto_reduce_npol).compress()

            else:
                raise NotImplementedError
                # reorder from pola_s, pola_i, m, mu_s, mu_i to  m, mu_s, pola_s, mu_i, pola_i
                # mat = np.moveaxis(self.values, (0, 1), (2, 4)) # 0 becomes 2, 1 becomes 4
                # merge mu_s * pola_s and mu_i * pola_i
                # return smrt_matrix(np.reshape(mat, (mat.shape[0], mat.shape[1]*mat.shape[2],
                # mat.shape[3]*mat.shape[4])), mtype="compressed3")

        elif self.mtype == "dense4":
            if self.values.shape[0] == 3 and auto_reduce_npol and mode == 0:
                # 3pol->2pol
                mat = self.values[0:2, 0:2, :, :]
            else:
                mat = self.values

            # reorder from pola_s, pola_i, mu_s, mu_i to  mu_s, pola_s, mu_i, pola_i
            assert(len(mat.shape) == 4)
            mat = np.moveaxis(mat, (0, 1), (1, 3))  # 0 becomes 1, 1 becomes 3, so 2 becomes 0 and 3 becomes 2
            # merge mu_s * pola_s and mu_i * pola_i
            return np.reshape(mat, (mat.shape[0] * mat.shape[1], mat.shape[2] * mat.shape[3]))  # return an 2x2 array !

        elif self.mtype == "diagonal5":
            if mode is not None:
                return self.sel(mode=mode, auto_reduce_npol=auto_reduce_npol).compress()
            else:
                raise NotImplementedError

        elif self.mtype == "diagonal4":
            if self.values.shape[0] == 3 and auto_reduce_npol and mode == 0:
                # 3pol->2pol
                mat = self.values[0:2, :]
            else:
                mat = self.values
            # reorder from pola, mu to mu*pola and compress
            assert(len(mat.shape) == 2)
            return smrt_diag(np.reshape(np.transpose(mat), mat.shape[0] * mat.shape[1]))

        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return smrt_matrix(other * self.values)

    def __mul__(self, other):
        return smrt_matrix(self.values * other)

    def __truediv__(self, other):
        return smrt_matrix(self.values / other)

    def __add__(self, other):
        if isinstance(other, smrt_matrix):
            return smrt_matrix(other.values + self.values)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, smrt_matrix):
            return smrt_matrix(other.values - self.values)
        else:
            raise NotImplementedError

    def __abs__(self):
        return np.abs(self.values)

    def __getitem__(self, key):
        if self.mtype == "0":
            return np.float64(0.)  # 0, but can be used as a numpy thing

        else:
            return self.values[key]

    def __setitem__(self, key, v):
        self.values[key] = v

    @property
    def diagonal(self):
        if self.mtype == "0":
            return np.array([[0.]])
        if self.mtype.startswith("diagonal"):
            return self.values
        else:
            return np.moveaxis(np.diagonal(np.diagonal(self.values, axis1=-2, axis2=-1)), -1, 0)  # diagonal in incidence angle and pola
            # the moveaxis is necessary to put back the pola indice at the first position because diagonal move the diagonale "index" to the end of the array.

    def sel(self, **kwargs):

        if 'mode' in kwargs:
            mode = kwargs['mode']
            # 3pol->2pol
            if self.values.shape[0] == 3 and kwargs['auto_reduce_npol'] and mode == 0:
                pola = slice(0, 2)
            else:
                pola = slice(None)
            if self.mtype == "dense5":
                return smrt_matrix(self.values[pola, pola, mode, :, :], mtype='dense4')
            elif self.mtype == "diagonal5":
                return smrt_matrix(self.values[pola, mode, :], mtype='diagonal4')

            elif self.mtype == "dense4":
                raise SMRTError("Dense4 matrix can not be selected by mode")

            elif self.mtype == "diagonal4":
                raise SMRTError("Diagonal4 matrix can not be selected by mode")
            else:
                raise NotImplementedError
        else:
            raise SMRTError("Currently only selection by mode is implemented")

    def __repr__(self):

        shape = getattr(self.values, "shape", "")
        return str("smrt_matrix %s %s" % (self.mtype, shape)) + "\n" + str(self.values)


def is_zero_scalar(m):
    return np.isscalar(m) and (m == 0)


def is_equal_zero(m):
    """return true if the smrt matrix is null"""

    if isinstance(m, smrt_diag):
        m = m.diagonal()

    return is_zero_scalar(m) or \
        (getattr(m, "mtype", None) == "0") or \
        (not np.any(m))


if numba:
    @numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)], cache=True)
    def abs2(x):
        return x.real**2 + x.imag**2
else:
    def abs2(x):
        return x.real**2 + x.imag**2


def generic_ft_even_matrix(phase_function, m_max, nsamples=None):
    """ Calculation of the Fourier decomposed of the phase or reflection or transmission matrix provided by the function.

    This method calculates the Fourier decomposition modes and return the output.

    Coefficients within the phase function are

    Passive case (m = 0 only) and active (m = 0) ::

        M  = [Pvvp  Pvhp]
             [Phvp  Phhp]

    Active case (m > 0)::

        M =  [Pvvp Pvhp Pvup]
             [Phvp Phhp Phup]
             [Puvp Puhp Puup]

    :param phase_function: must be a function taking dphi as input. It is assumed that phi is symmetrical (it is in cos(phi))
    :param m_max: maximum Fourier decomposition mode needed

    """

    # samples of dphi for fourier decomposition. Highest efficiency for 2^n. 2^2 ok
    if nsamples is None:
        nsamples = 2**np.ceil(3 + np.log(m_max + 1) / np.log(2))

    assert nsamples > 2 * m_max

    # dphi must be evenly spaced from 0 to 2 * np.pi (but not including period), but we can use the symmetry of the phase function
    # to reduce the computation to 0 to pi (including 0 and pi) and mirroring for pi to 2*pi (excluding both)

    dphi = np.linspace(0, np.pi, int(nsamples // 2 + 1))

    # compute the phase function
    p = phase_function(dphi)

    npol = p.npol

    # mirror the phase function
    assert len(p.values.shape) == 5
    p_mirror = p.values[:, :, -2:0:-1, :, :].copy()
    if npol >= 3:
        p_mirror[0:2, 2] = -p_mirror[0:2, 2]
        p_mirror[2, 0:2] = -p_mirror[2, 0:2]

    # concatenate the two mirrored phase function
    p = np.concatenate((p.values, p_mirror), axis=2)
    assert(p.shape[2] == nsamples)

    # compute the Fourier Transform of the phase function along phi axis (axis=2)
    ft_p = np.fft.fft(p, axis=2)

    ft_even_p = smrt_matrix.empty((npol, npol, m_max + 1, p.shape[-2], p.shape[-1]))

    # m=0 mode
    ft_even_p[:, :, 0] = ft_p[:, :, 0].real * (1.0 / nsamples)

    # m>=1 modes
    if npol == 2:
        # the factor 2 comes from the change exp -> cos, i.e. exp(-ix) + exp(+ix)= 2 cos(x)
        ft_even_p[:, :, 1:] = ft_p[:, :, 1:m_max + 1].real * (2.0 / nsamples)

    else:
        delta = 2.0 / nsamples
        ft_even_p[0:2, 0:2, 1:] = ft_p[0:2, 0:2, 1:m_max + 1].real * delta

        # For the even matrix:
        # Sin components needed for p31, p32. Negative sin components needed for p13, p23. Cos for p33
        # The sign for 0:2, 2 and 2, 0:2 have been double check with Rayleigh and Mazter 2006 formulation of the Rayeligh Matrix (p111-112)
        ft_even_p[0:2, 2, 1:] = ft_p[0:2, 2, 1:m_max + 1].imag * delta
        ft_even_p[2, 0:2, 1:] = - ft_p[2, 0:2, 1:m_max + 1].imag * delta
        ft_even_p[2, 2, 1:] = ft_p[2, 2, 1:m_max + 1].real * delta

    return ft_even_p  # order is pola_s, pola_i, m, mu_s, mu_i


def set_max_numerical_threads(nthreads):
    """set the maximum number of threads for a few known library. This is useful to disable parallel computing in 
SMRT when using parallel computing to call multiple // SMRT runs. This avoid over-committing the CPUs and results 
in much better performance. Inspire from joblib."""

    nthreads = str(nthreads)
    os.environ['MKL_NUM_THREADS'] = nthreads
    os.environ['OPENBLAS_NUM_THREADS'] = nthreads
    os.environ['OMP_NUM_THREADS'] = nthreads
    os.environ['VECLIB_MAXIMUM_THREADS'] = nthreads
    os.environ['NUMEXPR_NUM_THREADS'] = nthreads


def cached_roots_legendre(n):
    """
    Cache roots_legendre results to speed up calls of the fixed_quad
    function.
    """
    if n in cached_roots_legendre.cache:
        return cached_roots_legendre.cache[n]

    from scipy.special import roots_legendre

    cached_roots_legendre.cache[n] = roots_legendre(n)
    return cached_roots_legendre.cache[n]


cached_roots_legendre.cache = dict()
