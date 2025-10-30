import numpy as np
import numpy.typing as npt
from smrt.core.globalconstants import C_SPEED
from . import multifresnel

def compute_matrix_slab_derivatives(
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
        L, (mu2, tau_layer) = multifresnel.forward_matrix_fulloutput(
            eps_1,
            eps_2,
            mu,
            kd=kd_,
            temperature=temperature_,
            limit_optical_depth=tau_max,
        )

        M = multifresnel.matmul3(M, L) if M is not None else L
        tau_max -= tau_layer  # decrease the optical depth
        tau_snowpack += tau_layer[imumax]

        if tau_max[imumax] < 0:
            break

        mu = mu2
        eps_1 = eps_2
    dM = [M for _ in thickness]
    return M, dM, tau_snowpack