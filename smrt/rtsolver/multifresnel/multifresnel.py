"""
This code computes the reflection for a stack of multiple incoherent layers.
A convenience function is provided to compute the reflection of a stack of pair of layers

References:
https://hal.science/hal-01155614/document

https://arxiv.org/pdf/1603.02720.pdf
"""

from collections.abc import Sequence
from typing import Optional

import numpy as np
import numpy.typing as npt
from numba import jit

from smrt.core import fresnel
from smrt.core.globalconstants import C_SPEED

# compile fresnel_coefficients
fresnel_coefficients_maezawa09_rigorous = jit(nopython=True, cache=True)(
    fresnel.fresnel_coefficients_maezawa09_rigorous
)

VPOL = 0
HPOL = 1


def compute_matrix_slab(
    frequency: float,
    outmu: npt.NDArray[np.floating],
    permittivity: npt.NDArray[complex],
    temperature: npt.NDArray[np.floating],
    thickness: npt.NDArray[np.floating],
    prune_deep_snowpack: int = 10,
    air_permittivity: int = 1,
):
    """Compute the transfer matrix for a slab of layers.

    Args:
        frequency: frequency
        outmu: cosine angle in the medium above the slab (usually air)
        permittivity: permittivity of the layers
        temperature: temperature of the layers
        thickness: thickness of the layers
        prune_deep_snowpack: this value is the optical depth from which the layers are discarded in the calculation.
            It is to be use to accelerate the calculations for deep snowpacks or at high frequencies when the
            contribution of the lowest layers is neglegible.
        air_permittivity: permittivity in above the slab
    """

    kd = 2 * np.pi * frequency / C_SPEED * np.array(thickness)

    mu = np.atleast_1d(outmu)
    imumax = np.argmax(mu)
    tau_max = np.full_like(mu, prune_deep_snowpack)

    eps_1 = air_permittivity

    M = None
    tau_snowpack = 0.0
    for eps_2, temperature_, kd_ in zip(permittivity, temperature, kd):
        L, (mu2, tau_layer) = forward_matrix_fulloutput(
            eps_1,
            eps_2,
            mu,
            kd=kd_,
            temperature=temperature_,
            limit_optical_depth=tau_max,
        )

        M = matmul3(M, L) if M is not None else L

        tau_max -= tau_layer  # decrease the optical depth
        tau_snowpack += tau_layer[imumax]

        if tau_max[imumax] < 0:
            break

        mu = mu2
        eps_1 = eps_2
    return M, tau_snowpack


def forward_matrix(
    eps1: complex, eps2: complex, mu: float, kd: float, temperature: float, limit_optical_depth: Optional[float] = None
):
    """compute the operator to go from layer 1 to 2.

    Args:
        eps1: permittivity of the upper layer
        eps2: permittivity of the lower layer
        mu: cosine angle
        kd: layer thickness multipled by the wavenumber
        temperature: layer temperature
        limit_optical_depth: optional alue that limit the optical depth of a layer

    Results:
        return the matrix
    """
    return forward_matrix_fulloutput(eps1, eps2, mu, kd, temperature, limit_optical_depth=limit_optical_depth)[0]


@jit(nopython=True, cache=True)
def forward_matrix_fulloutput(
    eps1: complex, eps2: complex, mu: float, kd: float, temperature: float, limit_optical_depth: Optional[float] = None
):
    """compute the operator to go from layer 1 to 2.

    Args:
        eps1: permittivity of the upper layer
        eps2: permittivity of the lower layer
        mu: cosine angle
        kd: layer thickness multipled by the wavenumber
        temperature: layer temperature
        limit_optical_depth: optional alue that limit the optical depth of a layer

    Results:
        return the matrix along with the cosine angle in the lower layer and the optical depth
    """
    mu = np.atleast_1d(mu)
    # mu is above the current layer

    rv, rh, mu2 = fresnel_coefficients_maezawa09_rigorous(eps1, eps2, mu)

    # r = np.array([abs2(rv), abs2(rh)])
    r = np.stack((rv.real**2 + rv.imag**2, rh.real**2 + rh.imag**2))

    optical_depth = 2 * np.sqrt(eps2).imag * kd / mu2
    if limit_optical_depth is not None:
        optical_depth = np.clip(optical_depth, 0, limit_optical_depth)
    trans_v = np.exp(-optical_depth)  # power attenuation
    trans_v = trans_v[None, :]  # same as P = np.stack((P, P))

    # matrix layer:
    l13 = -(1 / trans_v - 1) * temperature
    l23 = (1 - trans_v) * temperature

    # product of the interface matrix and layer matrix
    # see the equation in overlead projet "multi-fresnel"
    M = np.empty((2, 3, 2, len(mu)))

    M[0, 0] = 1 / trans_v
    M[0, 1] = -r * trans_v
    M[0, 2] = l13 - r * l23

    M[1, 0] = r / trans_v
    M[1, 1] = (1 - 2 * r) * trans_v
    M[1, 2] = r * l13 + (1 - 2 * r) * l23

    # M[2, 0] = 0
    # M[2, 1] = 0
    # M[2, 2] = 1

    M /= 1 - r

    # return M and layer optical depth
    return M, (mu2, optical_depth)


@jit(nopython=True, cache=True)
def combine(Fs: Sequence):
    """Combine several layers by performing the matrix multiplications.

    Args:
        Fs: sequence of matrices
    """

    Fout = Fs[0]

    for F in Fs[1:]:
        # axes = [(0, 1), (0, 1), (0, 1)]  # assume the first two dimensions are the matrix dimension.
        Fout = matmul3(Fout, F)
    return Fout


@jit(nopython=True, cache=True)
def matrix_power(a: npt.NDArray[np.floating], n: int):
    """Compute the power of a matrix using an efficient binary decomposition.
    Taken from matrix_power in numpy but it uses the matmul3 function to perform the multiplications

    Args:
        a: matrix
        n: exponent (positive integer)
    """

    # axes = [(0, 1), (0, 1), (0, 1)]  # assume the first two dimensions are the matrix dimension.
    assert n >= 3

    z = result = None

    while n > 0:
        # z = a if z is None else np.matmul(z, z, axes=axes)
        z = a if z is None else matmul3(z, z)
        n, bit = divmod(n, 2)
        if bit:
            # result = z if result is None else np.matmul(result, z, axes=axes)
            result = z if result is None else matmul3(result, z)

    return result


def compute_emerging_radiation(M: npt.NDArray[np.floating]):
    """Compute the emerging radiation from the slab of layers.

    Args:
        M: the transfer matrix
    """

    return -M[1, 0] * M[0, 2] / M[0, 0] + M[1, 2]


def compute_reflection(M: npt.NDArray[np.floating]):
    """Compute the reflection from the slab of layers. This requires a matrix calculated with T=0

    Args:
        M: the transfer matrix
    """

    return M[1, 0, VPOL, :] / M[0, 0, VPOL, :], M[1, 0, HPOL, :] / M[0, 0, HPOL, :]


@jit(nopython=True, cache=True)
def matmul3(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]):
    """Compute the matrix multiplication considering the special shape of the matrices.

    The matrices are of the form
    (a00 a01 a02)
    (a11 a11 a12)
    (0 0 1)

    The last raw is implicit, it is not stored.

    Args:
        a: first matrix
        b: second matrix
    """

    assert a.shape == b.shape

    c = np.empty((2, 3, *a.shape[2:]), dtype=a.dtype)

    c[0, :] = a[0, 0] * b[0, :] + a[0, 1] * b[1, :]
    c[1, :] = a[1, 0] * b[0, :] + a[1, 1] * b[1, :]

    c[:, 2] += a[:, 2]

    return c
