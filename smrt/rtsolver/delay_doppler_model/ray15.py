"""This module calculates Delay Doppler Maps (DDM) based on Ray et al. 2015 with a semi-analytical solution.

References:
    Ray, C., Martin-Puig, C., Clarizia, M. P., Ruffini, G., Dinardo, S., Gommenginger, C., & Benveniste, J. (2015). SAR
    Altimeter Backscattered Waveform Model. IEEE Transactions on Geoscience and Remote Sensing, 53(2), 911–919.
    https://doi.org/10.1109/tgrs.2014.2330423
"""

import numba
import numpy as np
import numpy.typing as npt
import scipy.integrate

from smrt.core.error import SMRTError
from smrt.core.globalconstants import C_SPEED, LOG2
from smrt.core.terrain import TerrainInfo

from .delay_doppler_utils import (
    check_low_ndoppler,
    delay_sampling_vector,
    doppler_frequency_vector,
    ptr_gaussian_approximation,
)


class Ray15(object):
    """Build a delay Doppler map model Ray15.

    Args:
        sensor: SAR altimetry sensor object.
        oversampling_time: oversampling factor in time dimension.
        oversampling_doppler: oversampling factor in Doppler dimension.
        ptr_time: shape of the point target response in time dimension.
        ptr_doppler: shape of the point target response in Doppler dimension.
    """

    def __init__(
        self,
        sensor,
        oversampling_time: int = 4,
        oversampling_doppler: int = 4,
        ptr_time: str = "gaussian",
        ptr_doppler: str = "gaussian-smrt",
    ):
        super().__init__()

        self.sensor = sensor
        check_low_ndoppler(sensor.ndoppler, ddm_model=self)
        self.oversampling_time = oversampling_time
        self.oversampling_doppler = oversampling_doppler

        # R15 Eq 14
        self.Lx = (
            (C_SPEED * sensor.altitude * sensor.pulse_repetition_frequency)
            / (2 * sensor.velocity * sensor.ndoppler * sensor.frequency)
        )  # along-track # fmt: skip

        # R15 Eq 21  # in R15, tau_u is the useable length of the pulse and s the chirp slope. It appears that s*tau_u =
        # effective bandwidth = sensor.pulse_bandiwth
        self.Ly = np.sqrt(C_SPEED * sensor.altitude / (sensor.alpha * sensor.pulse_bandwidth))  # cross-track
        self.Lz = C_SPEED / (2 * sensor.pulse_bandwidth)

        beamwidth = (sensor.beamwidth_alongtrack + sensor.beamwidth_acrosstrack) / 2  # get a 'circular' antenna pattern
        self.gamma = 2 / LOG2 * np.sin(np.deg2rad(beamwidth) / 2) ** 2

        assert ptr_time.startswith("gaussian")  # R15 uses the gaussian approx. It is not explicit though...
        self.sigma_g, self.Ag = ptr_gaussian_approximation(ptr_doppler, sensor.doppler_window)

        # R15 Eq 37
        self.K = (
            sensor.wavelength**2
            * sensor.ndoppler**2
            * self.Lx
            * self.Ly
            / (4 * np.pi * sensor.altitude**4)
            * np.sqrt(2 * np.pi)
            * self.Ag**2
            * self.sigma_g**2
        )  # fmt: skip  # unit: no unit

        self.tau = delay_sampling_vector(sensor, oversampling_time)
        self.fdoppler = doppler_frequency_vector(sensor, oversampling_doppler)

    def delay_doppler_map(self, terrain_info: TerrainInfo):
        """Compute the delay Doppler map for a surface.

        Note that Ray 2015 is formulated for a free function but is implemented for a constant backscatter only to cope
        with the optimization with numba.
        """
        if terrain_info.distribution != "normal":
            raise SMRTError("Ray15 is only available for a normally distributed surface")

        sensor = self.sensor

        l = self.fdoppler[np.newaxis, :] * sensor.ndoppler / sensor.pulse_repetition_frequency  # doppler beam
        k = self.tau[:, np.newaxis] * sensor.pulse_bandwidth  # time gate

        # R15 Eq 38
        k_off = 0  # (mean_z - Zem) / Lz
        Kappa = k + k_off

        # R15 Eq 37
        sigma_s = terrain_info.sigma_surface / self.Lz
        # R15 Eq 39  # the middle term explain the X shape of the DDM.
        gl = 1 / np.sqrt(self.sigma_g**2 + (2 * self.sigma_g * l * self.Lx**2 / self.Ly**2) ** 2 + sigma_s**2)

        def BTmaps(k, l):
            args = (
                k,
                l,
                terrain_info.sigma_surface,
                self.Lx,
                self.Ly,
                self.Lz,
                sensor.altitude,
                self.gamma * sensor.alpha,
                sensor.antenna_gain,
            )

            if sigma_s < 0.2:  # small roughness. No need to integrate
                Bkl = B_integrand(0, k, l, 0.2 * self.Lz, *args[3:])
                Tkl = 0
            else:
                coefB = 1 / (np.sqrt(2 * np.pi) * terrain_info.sigma_surface)
                Bkl = scipy.integrate.quad(B_integrand, -np.inf, np.inf, args=args)[0] * coefB

                coefT = self.Lz / (Bkl * terrain_info.sigma_surface**2) * coefB
                Tkl = scipy.integrate.quad(T_integrand, -np.inf, np.inf, args=args)[0] * coefT
            return Bkl, Tkl

        # Eq 41
        Bkl, Tkl = np.vectorize(BTmaps)(k, l)

        Pkl = (
            self.K
            # * sigma0
            * Bkl
            * np.sqrt(gl)
            * ((1 + Bkl * k_off) * fn(gl * Kappa, n=0) + Tkl * gl * sigma_s**2 * fn(gl * Kappa, n=1))
        )
        return Pkl  # unit: no unit


@numba.jit(nopython=True, cache=True)
def fn_integrand(v: npt.NDArray, n: int, xi: npt.NDArray):
    # Eq 40
    return (v**2 - xi) ** n * np.exp(-((v**2 - xi) ** 2) / 2)


def fn(xi: npt.NDArray, n: int):
    # from 0 to +infty
    # a, b = 0, np.inf

    # in reality it is enough to integrate around sqrt(xi)
    delta = np.sqrt(2 * 50)

    return np.array(
        [scipy.integrate.quad(fn_integrand,
                              a=np.sqrt(max(0, xi_ - delta)), b=np.sqrt(max(0, xi_ + delta)),
                              args=(n, xi_, ))[0] for xi_ in np.atleast_1d(xi)]
    )  # fmt: skip


fn = np.vectorize(fn)


@numba.jit(nopython=True)
def Gamma(
    x: npt.NDArray, y: npt.NDArray, altitude: npt.NDArray, gamma_alpha: npt.NDArray, G0: npt.NDArray
) -> npt.NDArray:
    # Brown model for the antenna gain and assuming a constant backscatter

    rho2 = x**2 + y**2
    rho = np.sqrt(rho2)

    rho_h = rho / altitude  # DEF: rho_h = np.tan(theta)

    # from Brown with misspointting/off_nadir_angle
    # cosphi = ...
    # newtheta = np.arccos((np.cos(self.sensor.off_nadir_angle) + rho_h * np.sin(self.sensor.off_nadir_angle)
    #                              * np.cos(phi)) / np.sqrt(1 + rho_h**2))
    # from Brown without misspointting/off_nadir_angle
    newtheta = np.arccos(1 / np.sqrt(1 + rho_h**2))  # without off_nadir_angle

    # return G**2
    return G0**2 * np.exp(-2 * 2 / (gamma_alpha) * np.sin(newtheta) ** 2)  # from brown_1977


@numba.jit(nopython=True)
def Gamma_e(
    x: npt.NDArray, y: npt.NDArray, altitude: npt.NDArray, gamma_alpha: npt.NDArray, G0: npt.NDArray
) -> npt.NDArray:
    return Gamma(x, y, altitude, gamma_alpha, G0) + Gamma(x, -y, altitude, gamma_alpha, G0)


@numba.jit(nopython=True)
def Gamma_kl(
    k: npt.NDArray,
    l: npt.NDArray,
    z: npt.NDArray,
    Lx: npt.NDArray,
    Ly: npt.NDArray,
    Lz: npt.NDArray,
    altitude: npt.NDArray,
    gamma_alpha: npt.NDArray,
    G0: npt.NDArray,
) -> npt.NDArray:
    return Gamma_e(Lx * l, Ly * np.sqrt(np.maximum(k + z / Lz, 0)), altitude, gamma_alpha, G0)


@numba.jit(nopython=True)
def B_integrand(
    z: npt.NDArray,
    k: npt.NDArray,
    l: npt.NDArray,
    sigma_z: npt.NDArray,
    Lx: npt.NDArray,
    Ly: npt.NDArray,
    Lz: npt.NDArray,
    altitude: npt.NDArray,
    gamma_alpha: npt.NDArray,
    G0: npt.NDArray,
) -> npt.NDArray:
    # R15 Eq 32
    return np.exp(-((z / sigma_z) ** 2) / 2) * Gamma_kl(k, l, z, Lx, Ly, Lz, altitude, gamma_alpha, G0)


@numba.jit(nopython=True)
def T_integrand(
    z: npt.NDArray,
    k: npt.NDArray,
    l: npt.NDArray,
    sigma_z: npt.NDArray,
    Lx: npt.NDArray,
    Ly: npt.NDArray,
    Lz: npt.NDArray,
    altitude: npt.NDArray,
    gamma_alpha: npt.NDArray,
    G0: npt.NDArray,
) -> npt.NDArray:
    # R15 Eq 32
    return np.exp(-((z / sigma_z) ** 2) / 2) * z * Gamma_kl(k, l, z, Lx, Ly, Lz, altitude, gamma_alpha, G0)
