"""This modules provide utils for to compute the streams"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import scipy.optimize

from smrt.core.error import SMRTError, smrt_warn
from smrt.core.lib import cached_roots_legendre

#
# Compute streams with different method.
#


def make_empty_array():
    return np.ndarray([])


@dataclass
class Streams(object):
    n: List[int] = field(default_factory=list)
    mu: List[np.ndarray] = field(default_factory=list)
    weight: List[np.ndarray] = field(default_factory=list)
    outmu: np.ndarray = field(default_factory=make_empty_array)
    outweight: np.ndarray = field(default_factory=make_empty_array)
    # n_substrate: int = 0
    n_air: int = 0


def compute_stream(n_max_stream, permittivity, mode="most_refringent") -> Streams:
    # """Compute the optimal angles of each layer. Use for this a Gauss-Legendre quadrature for the most refringent layer and
    # use Snell-law to prograpate the direction in the other layers takig care of the total reflection.

    #     :param n_max_stream: number of stream
    #     :param permittivity: permittivity of each layer
    #     :type permittivity: ndarray
    #     :returns: mu, weight, outmu
    # """

    if mode in ["most_refringent", "air"]:
        return compute_stream_gaussian(n_max_stream, permittivity, mode=mode)

    elif mode == "uniform_air":
        return compute_stream_uniform(n_max_stream, permittivity)

    else:
        raise SMRTError(f"Unknown mode '{mode}' for the computation of the streams")


def compute_stream_gaussian(n_max_stream, permittivity, mode="most_refringent"):
    # """Compute the optimal angles of each layer. Use for this a Gauss-Legendre quadrature for the most refringent layer and
    # use Snell-law to prograpate the direction in the other layers takig care of the total reflection.

    #     :param n_max_stream: number of stream
    #     :param permittivity: permittivity of each layer
    #     :type permittivity: ndarray
    #     :returns: mu, weight, outmu
    # """

    nlayer = len(permittivity)

    if nlayer == 0:
        outmu, outweight = gauss_legendre_quadrature(n_max_stream)
        return Streams(n_air=n_max_stream, outmu=outmu, outweight=outweight)

    # there are some layers

    #  ### search and proceed with the most refringent layer
    k_most_refringent = np.argmax(permittivity)
    real_index_air = np.real(np.sqrt(permittivity[k_most_refringent] / 1.0))

    streams = Streams()

    if mode is None or mode == "most_refringent":
        # calculate the gaussian weights and nodes for the most refringent layer
        mu_most_refringent, weight_most_refringent = gauss_legendre_quadrature(n_max_stream)

    elif mode == "air":
        smrt_warn("This code has not been tested yet. Use with caution.")

        def number_stream_in_air(n_stream_densest_layer):
            mu_most_refringent, weight_most_refringent = gauss_legendre_quadrature(int(n_stream_densest_layer))
            relsin = real_index_air * np.sqrt(1 - mu_most_refringent**2)
            return np.sum(relsin < 1) - n_max_stream

        streams.n = int(scipy.optimize.brentq(number_stream_in_air, n_max_stream / 4, n_max_stream * 4))
        mu_most_refringent, weight_most_refringent = gauss_legendre_quadrature(streams.n)

    else:
        raise RuntimeError("Unknow mode to compute the number of stream")

    #  calculate the nodes and weights of all the other layers

    # calculate real part of the index. It is an approximation.
    # See for instance "Semiconductor Physics, Quantum Electronics & Optoelectronics. 2001. V. 4, N 3. P. 214-218."
    real_index = np.real(np.sqrt(permittivity[k_most_refringent] / permittivity[:]))

    # calculate the angles (=node)
    relsin = real_index[:, np.newaxis] * np.sqrt(1 - mu_most_refringent[np.newaxis, :] ** 2)

    real_reflection = relsin < 1  # mask where real reflection occurs

    mu = np.zeros((nlayer, n_max_stream), dtype=np.float64)
    mu[real_reflection] = np.sqrt(1 - relsin[real_reflection] ** 2)

    # calculate the number of streams per layer
    streams.mu = [mu[l, real_reflection[l, :]] for l in range(nlayer)]
    streams.n = np.sum(real_reflection, axis=1)

    assert all(np.size(n) for n in streams.n)

    # calculate the weight ("a" in Y-Q Jin)
    # weight(1,k)=1-0.5 * (mu(1,k)+mu(2,k))
    # weight(nsk,k)=0.5 * (mu(nsk-1,k)+mu(nsk,k))
    # weight(2:nsk-1,k)=0.5 * (mu(1:nsk-2,k)-mu(3:nsk,k))

    streams.weight = compute_weight(streams.mu)

    # ### calculate the angles (=node) in the air
    # real_index = np.real(np.sqrt(permittivity[0]/1.0))
    # relsin = real_index * np.sqrt(1 - mu[0, :]**2)

    # real_reflection = relsin < 1
    # outmu = np.sqrt(1 - relsin[real_reflection]**2)

    relsin = real_index_air * np.sqrt(1 - mu_most_refringent[:] ** 2)

    real_reflection = relsin < 1
    streams.outmu = np.sqrt(1 - relsin[real_reflection] ** 2)
    streams.n_air = len(streams.outmu)

    streams.outweight = compute_outweight(streams.outmu)

    # compute the number of stream in the substrate
    # streams.n_substrate = compute_n_stream_substrate(permittivity, substrate_permittivity, streams.mu)

    return streams


def compute_stream_uniform(n_max_stream, permittivity):
    # """Compute the angles of each layer. Use a regular step in angle in the air, then deduce the angles in the other layers
    # using Snell-law. Then, in the most refringent layer, add regular stream up to close to 0, and then propagate back this second
    # set of angles in the other layers using Snell-law and accounting for the total reflections

    #     :param n_max_stream: number of stream
    #     :param permittivity: permittivity of each layer
    #     :type permittivity: ndarray
    #     :returns: mu, weight, outmu
    # """

    #
    # first set has uniform angle distribution in the air
    #
    streams = Streams(n_air=n_max_stream, outmu=np.cos(np.linspace(0.01, np.pi / 2 * 0.99, n_max_stream)))

    nlayer = len(permittivity)

    if nlayer == 0:
        streams.outweight = compute_outweight(streams.outmu)
        return streams

    #  calculate the nodes and weights of all the other layers

    # calculate real part of the index. It is an approximation.
    # See for instance "Semiconductor Physics, Quantum Electronics & Optoelectronics. 2001. V. 4, N 3. P. 214-218."
    real_index = np.real(np.sqrt(1 / permittivity[:]))

    # calculate the angles (=node)
    relsin = real_index[:, np.newaxis] * np.sqrt(1 - streams.outmu[np.newaxis, :] ** 2)

    # deduce the first set of streams
    mu1 = np.sqrt(1 - relsin**2)

    # now compute the additional streams.
    # get the most_refringent layer
    k_most_refringent = np.argmax(permittivity)

    # compute the mean mu resolution to extend the first set
    mean_resolution = np.mean(np.diff(mu1[k_most_refringent]))

    # compute the other streams
    mu2_most_refringent = np.arange(mu1[k_most_refringent][-1], 0.02, mean_resolution)
    # calculate real part of the index. It is an approximation.
    # See for instance "Semiconductor Physics, Quantum Electronics & Optoelectronics. 2001. V. 4, N 3. P. 214-218."
    real_index = np.real(np.sqrt(permittivity[k_most_refringent] / permittivity[:]))

    # calculate the angles (=node)
    relsin = real_index[:, np.newaxis] * np.sqrt(1 - mu2_most_refringent[np.newaxis, :] ** 2)

    real_reflection = relsin < 1  # mask where real reflection occurs

    # compute the second set of angles
    mu2 = np.zeros((nlayer, len(mu2_most_refringent)), dtype=np.float64)
    mu2[real_reflection] = np.sqrt(1 - relsin[real_reflection] ** 2)

    # assemble the two sets
    streams.mu = [np.hstack((mu1[l], mu2[l, real_reflection[l, :]])) for l in range(nlayer)]
    # calculate the number of streams per layer
    streams.n = n_max_stream + np.sum(real_reflection, axis=1)

    assert all(np.size(n) > 2 for n in streams.n)

    # compute the weights
    streams.weight = compute_weight(streams.mu)
    streams.outweight = compute_outweight(streams.outmu)

    # compute the number of stream in the substrate
    # streams.n_substrate = compute_n_stream_substrate(permittivity, substrate_permittivity, streams.mu)

    return streams


def gauss_legendre_quadrature(n):
    """
    Return the gauss-legendre roots and weight, only the positive roots are return in descending order.

    Args:
        n: number of (positive) points in the quadrature. Must be larger than 2
    """

    assert n >= 2

    mu, weight = cached_roots_legendre(2 * n)

    mu = mu[-1 : n - 1 : -1]
    weight = weight[-1 : n - 1 : -1]

    return mu, weight


def compute_outweight(outmu):
    outweight = np.empty_like(outmu)
    outweight[0] = 1 - 0.5 * (outmu[0] + outmu[1])
    outweight[-1] = 0.5 * (outmu[-2] + outmu[-1])
    outweight[1:-1] = 0.5 * (outmu[0:-2] - outmu[2:])
    return outweight


def compute_weight(mu):
    weight = [np.empty_like(m) for m in mu]
    for l in range(len(mu)):
        weight[l][0] = 1 - 0.5 * (mu[l][0] + mu[l][1])
        weight[l][-1] = np.abs(0.5 * (mu[l][-2] + mu[l][-1]))
        weight[l][1:-1] = np.abs(0.5 * (mu[l][0:-2] - mu[l][2:]))
    return weight


def compute_n_stream_substrate(permittivity, substrate_permittivity, mu):
    if substrate_permittivity is None:
        n_substrate = None
    else:
        real_index = np.real(np.sqrt(substrate_permittivity / permittivity[-1]))

        # calculate the angles (=node) in the substrate

        # get the most_refringent layer
        k_most_refringent = np.argmax(permittivity)

        relsin = real_index * np.sqrt(1 - mu[k_most_refringent][:] ** 2)
        n_substrate = np.sum(relsin < 1)  # count where real reflection occurs

    return n_substrate
