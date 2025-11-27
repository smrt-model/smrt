import os
from collections.abc import Sequence
from typing import Type, Union

import numpy as np
import pandas as pd

from smrt.core.error import SMRTError
from smrt.core.optional_numba import numba


def get(x, i, name=None):
    """
    Return the i-th value in an array or dict of array. Can deal with scalar as well. In this case, it repeats the value.

    Args:
        x: flexible array like object or scalar
        i: i index to get
        name: name of the object x, for reporting error messages. Defaults to None.

    Raises:
        SMRTError: if x is too short compared to i

    Returns:
        : element from x
    """
    # function to take the i-eme value in an array or dict of array. Can deal with scalar as well. In this case, it repeats the value.

    if isinstance(x, str):
        return x
    elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        if i >= len(x.values):
            raise SMRTError(f"The array {name} is too short compared to the thickness array")
        return x.values[i]
    if isinstance(x, Sequence) or isinstance(x, np.ndarray):
        if i >= len(x):
            raise SMRTError(f"The array {name} is too short compared to the thickness array.")
        return x[i]
    elif isinstance(x, dict):
        return {k: get(x[k], i, k) for k in x}
    else:
        return x


def check_argument_size(x, n, name=None):
    """
    Check that x is either a scalar or a sequence of exactly n items and raise an error otherwise.

    Args:
        x: array like object or scalar
        n: expected size
        name: name of the object x, for reporting error messages. Defaults to None.
    Raises:
        SMRTError: if x is not a scalar or a sequence of size n
    """

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
        raise SMRTError(f"The array {name} must be a scalar or have the same size as the 'thickness' array.")


def is_sequence(x):
    """
    Check that x is a sequence

    Args:
        x: flexible object

    Returns:
        : True if x is a sequence, False otherwise
    """
    # maybe not the smartest way...
    return (
        isinstance(x, Sequence) or isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)
    ) and not isinstance(x, str)


def class_specializer(domain: str, cls: Union[str, Type], **options) -> Type:
    """
    Return a subclass of cls (imported from the domain if cls is a string) that use the provided "options" for __init__.

    This is equivalent to functools.partial but for a class.

    This is the same idea as:
    https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor
    """
    if isinstance(cls, str):
        from smrt.core.plugin import import_class  # lazy import

        cls = import_class(domain, cls)

    if not options:
        return cls

    def __init__(self, *args, **other_options):
        cls.__init__(self, *args, **options, **other_options)

    old_doc = getattr(cls, "__doc__")
    if old_doc is None:
        old_doc = "No original documentation."

    attributes = {
        "__init__": __init__,
        "__doc__": f"{old_doc}\n\nThis class was specialized with the following options: {options}",
        "__module__": cls.__module__,
    }
    new_name = f"Specialized {getattr(cls, '__name__', 'X')}"
    specialized_cls = type(new_name, (cls,), attributes)
    return specialized_cls


def len_atleast_1d(x):
    """
    Return length of x if it is an array or similar, otherwise return 1, or 0 if None.

    Args:
        x: object to return the length of

    Returns:
        : length of x
    """
    try:
        return len(x)
    except TypeError:
        return 1 if x is not None else 0


class smrt_diag(object):
    """
    Define a simple diagonal matrix class.

    Scipy.sparse is very slow for diagonal matrix and numpy has no good support for linear algebra. This diag class
    implements simple diagonal object without numpy subclassing (but without much features).
    It seems that proper subclassing numpy and overloading matmul is a very difficult problem.
    """

    __array_ufunc__ = None

    def __init__(self, arr):
        self.diag = np.atleast_1d(arr)
        assert (np.ndim(self.diag)) == 1

        # self.shape = shape if shape is not None else (len(self.diag), len(self.diag))

    def diagonal(self):
        return self.diag

    @property
    def shape(self):
        return (len(self.diag), len(self.diag))

    def __len__(self):
        return len(self.diag)

    def __rmatmul__(self, other):
        # assert other.shape[1] == self.shape[0]
        if other.ndim == 2:
            return other * self.diag[np.newaxis, :]
        elif other.ndim == 1:
            return other * self.diag
        else:
            raise NotImplementedError("multiplication with diag is only implemented for 1-d and 2-d ndarray")

    def __matmul__(self, other):
        # assert self.shape[1] == other.shape[0]
        if isinstance(other, smrt_diag):
            # return a diagonal object
            return smrt_diag(other.diag * self.diag)  # , shape=(self.shape[0], other.shape[1]))
        elif other.ndim == 2:
            # other must be an ndarray
            return self.diag[:, np.newaxis] * other
        elif other.ndim == 1:
            return other * self.diag
        else:
            raise NotImplementedError("multiplication with diag is only implemented for 1-d and 2-d ndarray")

    def __rmul__(self, other):
        assert np.isscalar(other)
        return smrt_diag(other * self.diag)

    def __mul__(self, other):
        assert np.isscalar(other)
        return smrt_diag(self.diag * other)

    def __add__(self, other):
        if isinstance(other, smrt_diag):
            return smrt_diag(other.diag + self.diag)
        elif other is None or (np.isscalar(other) and other == 0):
            return self
        elif isinstance(other, np.ndarray):
            assert other.shape == self.shape  # we do not allow broadcasting (not yet)...
            other = other.copy()
            other[np.diag_indices_from(other)] += self.diag
            return other
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, smrt_diag):
            return smrt_diag(other.diag - self.diag)
        elif other is None or (np.isscalar(other) and other == 0):
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
            raise IndexError(
                "The index of a diag object must be a tuple with two indices. See smrt.core.lib for the rational of this diag object."
            )
        return self.diag[i] if i == j else 0


class smrt_matrix(object):
    """
    Return a smrt_matrix object.

    SMRT uses two formats of matrix: one most suitable to implement emmodel where equations are different for each
    polarization and another one suitable for DORT computation where stream and polarization are collapsed in one
    dimension to allow matrix operation. In addition, the reflection and transmission matrix are often diagonal matrix,
    which needs to be handled because it saves space and allow much faster operations. This class implemented all these
    features.
    """

    def __init__(self, mat, mtype=None):
        if is_zero_scalar(mat):
            self.values = np.float64(0.0)  # 0, but can be used as a numpy thing
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

    @property
    def trace(self):
        return np.sum(self.diagonal)

    @property
    def meantrace(self):
        return np.mean(self.diagonal)

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
        """
        Compresses a matrix. This comprises several actions:
        1) select one mode, if relevant (dense5, and diagonal5).
        2) reduce the number of polarization from 3 to 2 if mode==0 and auto_reduce_npol=True.
        3) convert the format of the matrix to compressed numpy, involving a change of the dimension order (pola and streams are merged).
        """
        if self.mtype == "0":
            return np.float64(0.0)  # 0, but can be used as a numpy thing

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
            assert len(mat.shape) == 4
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
            assert len(mat.shape) == 2
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
        elif other is None:  # convenient hack, for disabling some part of the calculations
            return smrt_matrix(self)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, smrt_matrix):
            return smrt_matrix(other.values - self.values)
        elif other is None:  # convenient hack, for disabling some part of the calculations
            return smrt_matrix(self)
        else:
            raise NotImplementedError

    def __abs__(self):
        return np.abs(self.values)

    def __getitem__(self, key):
        if self.mtype == "0":
            return np.float64(0.0)  # 0, but can be used as a numpy thing

        else:
            return self.values[key]

    def __setitem__(self, key, v):
        self.values[key] = v

    @property
    def diagonal(self):
        if self.mtype == "0":
            return np.array([[0.0]])
        if self.mtype.startswith("diagonal"):
            return self.values
        else:
            return np.moveaxis(
                np.diagonal(np.diagonal(self.values, axis1=-2, axis2=-1)), -1, 0
            )  # diagonal in incidence angle and pola
            # the moveaxis is necessary to put back the pola indice at the first position because diagonal move the diagonale "index" to the end of the array.

    def to_dense(self):
        if self.mtype in ["dense5", "dense4"]:
            return self.copy()
        elif self.mtype == "diagonal4":
            pola, inc = self.values.shape

            mat = np.diagflat(self.values).reshape((pola, pola, inc, inc))
            return smrt_matrix(mat, mtype="dense4")

        elif self.mtype == "diagonal5":
            pola, mode, inc = self.values.shape

            # in numba with two loops it would be much faster
            mat = np.stack([np.diagflat(self.values[:, i, :]).reshape((pola, pola, inc, inc)) for i in range(mode)])
            mat = np.moveaxis(mat, 0, 2)

            return smrt_matrix(mat, mtype="dense5")
        else:
            raise NotImplementedError

    def sel(self, **kwargs):
        if "mode" in kwargs:
            mode = kwargs["mode"]
            # 3pol->2pol
            if self.values.shape[0] == 3 and kwargs["auto_reduce_npol"] and mode == 0:
                pola = slice(0, 2)
            else:
                pola = slice(None)
            if self.mtype == "dense5":
                return smrt_matrix(self.values[pola, pola, mode, :, :], mtype="dense4")
            elif self.mtype == "diagonal5":
                return smrt_matrix(self.values[pola, mode, :], mtype="diagonal4")

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
        return str(f"smrt_matrix {self.mtype} {shape}") + "\n" + str(self.values)


def is_zero_scalar(m):
    """
    Returns true if the object is a scalar equal to zero

    Args:
        m: object to test
    Returns:
        : True if m is a scalar equal to zero
    """
    return np.isscalar(m) and (m == 0)


def is_equal_zero(m):
    """
    Returns true if the smrt matrix is null

    Args:
        m: object to test
    Returns:
        : True if m is equal to zero
    """

    if isinstance(m, smrt_diag):
        m = m.diagonal()

    return is_zero_scalar(m) or (getattr(m, "mtype", None) == "0") or (not np.any(m))


if numba:

    @numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)], cache=True)
    def abs2(x):
        return x.real**2 + x.imag**2
else:

    def abs2(x):
        return x.real**2 + x.imag**2


def generic_ft_even_matrix(phase_function, m_max, nsamples=None):
    """
    Compute the Fourier transform of an even matrix.

    This matrix can be a phase function, reflection or transmission matrix.

    Coefficients within the phase function are

    Passive case (m = 0 only) and active (m = 0) ::

        M  = [Pvvp  Pvhp]
             [Phvp  Phhp]

    Active case (m > 0)::

        M =  [Pvvp Pvhp Pvup]
             [Phvp Phhp Phup]
             [Puvp Puhp Puup]

    Args:
        phase_function: must be a function taking dphi as input. It is assumed that phi is symmetrical (it is in cos(phi))
        m_max: maximum Fourier decomposition mode needed
        nsamples: number of samples to use for the Fourier decomposition. If None, it is automatically computed.
    """

    # samples of dphi for fourier decomposition. Highest efficiency for 2^n. 2^2 ok
    if nsamples is None:
        nsamples = 2 ** np.ceil(3 + np.log(m_max + 1) / np.log(2))

    assert nsamples > 2 * m_max

    # dphi must be evenly spaced from 0 to 2 * np.pi (but not including period), but we can use the symmetry of the phase function
    # to reduce the computation to 0 to pi (including 0 and pi) and mirroring for pi to 2*pi (excluding both)

    dphi = np.linspace(0, np.pi, int(nsamples // 2 + 1))

    # compute the phase function
    p = phase_function(dphi)

    npol = p.npol

    if len(p.values.shape) != 5:
        p = p.to_dense()

    # mirror the phase function
    assert len(p.values.shape) == 5
    p_mirror = p.values[:, :, -2:0:-1, :, :].copy()

    if npol >= 3:
        p_mirror[0:2, 2] = -p_mirror[0:2, 2]
        p_mirror[2, 0:2] = -p_mirror[2, 0:2]

    # concatenate the two mirrored phase function
    p = np.concatenate((p.values, p_mirror), axis=2)
    assert p.shape[2] == nsamples

    # compute the Fourier Transform of the phase function along phi axis (axis=2)
    ft_p = np.fft.fft(p, axis=2)

    ft_even_p = smrt_matrix.empty((npol, npol, m_max + 1, p.shape[-2], p.shape[-1]))

    #
    # m=0 mode
    ft_even_p[:, :, 0] = ft_p[:, :, 0].real * (1.0 / nsamples)

    #
    # m>=1 modes
    delta = 2.0 / nsamples  # the factor 2 comes from the change exp -> cos, i.e. exp(-ix) + exp(+ix)= 2 cos(x)

    if npol == 2:
        ft_even_p[:, :, 1:] = ft_p[:, :, 1 : m_max + 1].real * delta

    else:
        ft_even_p[0:2, 0:2, 1:] = ft_p[0:2, 0:2, 1 : m_max + 1].real * delta

        # For the even matrix:
        # Sin components needed for p31, p32. Negative sin components needed for p13, p23. Cos for p33
        # The sign for 0:2, 2 and 2, 0:2 have been double check with Rayleigh and Mazter 2006 formulation of the Rayeligh Matrix (p111-112)
        ft_even_p[0:2, 2, 1:] = ft_p[0:2, 2, 1 : m_max + 1].imag * delta
        ft_even_p[2, 0:2, 1:] = -ft_p[2, 0:2, 1 : m_max + 1].imag * delta
        ft_even_p[2, 2, 1:] = ft_p[2, 2, 1 : m_max + 1].real * delta

    return ft_even_p  # order is pola_s, pola_i, m, mu_s, mu_i


def vectorize_angles(mu_s, mu_i, dphi, compute_cross_product=True, compute_sin=True):
    """
    Return angular cosines and sinus with proper dimensions, ready for vectorized calculations.

    Args:
        mu_s: scattering cosine angle.
        mu_i: incident cosine angle.
        dphi: azimuth angle between the scattering and incident directions.
        compute_cross_product: if True perform the computation for all elements in mu_s, mu_i, dphi (cross product)
            and if False perform the computation for each successive configuration in mu_s, mu_i and dphi (they must
            have the same shape).

    Returns:
        vectorize angles
    """

    mu_s = np.atleast_1d(mu_s)
    mu_i = np.atleast_1d(mu_i)
    dphi = np.atleast_1d(dphi)

    if compute_cross_product:
        dphi = dphi[:, np.newaxis, np.newaxis]
        mu_s = mu_s[np.newaxis, :, np.newaxis]
        mu_i = mu_i[np.newaxis, np.newaxis, :]

    sin_i = np.sqrt(1.0 - mu_i**2) if compute_sin else np.nan
    sin_s = np.sqrt(1.0 - mu_s**2) if compute_sin else np.nan

    sinphi = np.sin(dphi)
    cosphi = np.cos(dphi)

    return mu_s, sin_s, mu_i, sin_i, cosphi, sinphi


def set_max_numerical_threads(nthreads):
    """
    Set the maximum number of threads for a few known library.

    This is useful to disable parallel computing in SMRT when using parallel computing to call multiple // SMRT runs.
    This avoid over-committing the CPUs and results in much better performance. Inspire from joblib.
    """

    nthreads = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = nthreads
    os.environ["OPENBLAS_NUM_THREADS"] = nthreads
    os.environ["OMP_NUM_THREADS"] = nthreads
    os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
    os.environ["NUMEXPR_NUM_THREADS"] = nthreads


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
