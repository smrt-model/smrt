import numpy as np
import numpy.typing as npt
from smrt.core.globalconstants import C_SPEED
from . import multifresnel
from typing import Optional


def compute_matrix_slab_derivatives(
    frequency: float,
    outmu: npt.NDArray[np.floating],
    permittivity: npt.NDArray[np.complexfloating],
    temperature: npt.NDArray[np.floating],
    thickness: npt.NDArray[np.floating],
    prune_deep_snowpack: int = 10,
    air_permittivity: int = 1,
):
    """Compute the transfer matrix for a slab of layers and its partial derivatives w.r.t. the temperature of each layer.

    Args:
        frequency: frequency
        outmu: cosine angle in the medium above the slab (usually air)
        permittivity: permittivity of the layers
        temperature: temperature of the layers
        thickness: thickness of the layers
        prune_deep_snowpack: this value is the optical depth from which the layers are discarded in the calculation.
            It is to be used to accelerate the calculations for deep snowpacks or at high frequencies when the
            contribution of the lowest layers is neglegible. Set to 0 to avoid pruning (not implemented yet).
        air_permittivity: permittivity inthe middle above the slab
    """

    kd = 2 * np.pi * frequency / C_SPEED * np.array(thickness)

    mu = np.atleast_1d(outmu)
    imumax = np.argmax(mu)
    tau_max = np.full_like(mu, prune_deep_snowpack)

    eps_1 = air_permittivity

    derivatives = np.zeros([len(thickness), 2, 3, 2, len(mu)])
    M = None

    index = 0
    tau_snowpack = 0.0

    N = np.empty((len(thickness),2, 3, 2, len(mu)))
    #Algorithm to be implemented.
    for eps_2, temperature_, kd_ in zip(permittivity, temperature, kd):
        L, (mu2, tau_layer) = multifresnel.forward_matrix_fulloutput(
            eps_1,
            eps_2,
            mu,
            kd=kd_,
            temperature=temperature_,
            limit_optical_depth=tau_max,
        )
        #print(f"L.shape = {L.shape}")
        # forward_matrix_derivative is jsut a proxy for now
        # not sure we need dmu2 or dtaulayer
        dL, (dmu2, dtau_layer) = forward_matrix_derivative(
            eps_1, eps_2, mu, kd=kd_, temperature=temperature_, limit_optical_depth=tau_max
        )
        #print(f"dL.shape = {dL.shape}")
        M = multifresnel.matmul3(M, L) if M is not None else L
        tau_max -= tau_layer  # decrease the optical depth
        tau_snowpack += tau_layer[imumax]

        if tau_max[imumax] < 0:
            break

        mu = mu2
        eps_1 = eps_2
        #proxy
        derivatives[index] = dL
        N[index] = dL + M
        index +=1
    dM = N
    return M, dM, tau_snowpack


def forward_matrix_derivative(
    eps1: complex, eps2: complex, mu: float, kd: float, temperature: float, limit_optical_depth: Optional[float] = None
):
    """compute the operator to go from layer 1 to 2. Is just a copy of multifresnel.fmfo for now

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

    rv, rh, mu2 = multifresnel.fresnel_coefficients_maezawa09_rigorous(eps1, eps2, mu)

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
