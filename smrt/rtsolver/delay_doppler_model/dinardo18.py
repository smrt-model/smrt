"""This module calculates Delay Doppler Map (DDM) based on Dinardo et al. 2018.

This DDM model extends Ray et al. 2015 with a fully analytical solution under the conditions that the surface is Gaussian and
the backscatter decreases according to Geometrical Optics, using analytical functions.

References:
    - Dinardo, S., Fenoglio-Marc, L., Buchhaupt, C., Becker, M., Scharroo, R., Joana Fernandes, M., & Benveniste, J.
      (2018). Coastal SAR and PLRM altimetry in German Bight and West Baltic Sea. Advances in Space Research, 62(6),
      1371–1404. https://doi.org/10.1016/j.asr.2017.12.018

    - Ray, C., Martin-Puig, C., Clarizia, M. P., Ruffini, G., Dinardo, S., Gommenginger, C., & Benveniste, J. (2015).
      SAR Altimeter Backscattered Waveform Model. IEEE Transactions on Geoscience and Remote Sensing, 53(2), 911–919.
      https://doi.org/10.1109/tgrs.2014.2330423

"""

from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.special

from smrt.core.error import SMRTError
from smrt.core.globalconstants import C_SPEED, LOG2
from smrt.core.terrain import TerrainInfo

from .delay_doppler_utils import (
    check_low_ndoppler,
    delay_sampling_vector,
    doppler_frequency_vector,
    ptr_gaussian_approximation,
)


class Dinardo18(object):
    """Build a delay Doppler map model Dinardo18.

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
        ptr_time: Optional[str] = None,
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

        # R15 Eq 21
        self.Ly = np.sqrt(C_SPEED * sensor.altitude / (sensor.alpha * sensor.pulse_bandwidth))  # cross-track
        self.Lz = C_SPEED / (2 * sensor.pulse_bandwidth)
        # Eq 21 show chirp_slope * pulse_duration but we know that
        # chirp_slope = sensor.pulse_bandwidth / usable_pulse_duration   # in Hz / s

        # D18 Eq 18
        self.theta_lim = self.Lz / (sensor.alpha * self.Lx)
        # same as self.Ly**2 / 2 / self.Lx / sensor.altitude, self.theta_lim)  , see Eq 15

        if ptr_time is None:
            ptr_time = "gaussian"

        assert ptr_time.startswith("gaussian")  # R15 uses the gaussian approx. It is not explicit though...
        sigma_g, Ag = ptr_gaussian_approximation(ptr_doppler, sensor.doppler_window)

        # alpha_p = sigma_g according to D18. See also R15 Eq 39
        self.sigma_p = sigma_g / self.sensor.pulse_bandwidth
        print(f"{sigma_g=} {self.sigma_p=}")

        # D18 Eq 27
        self.gamma_x = 8 * LOG2 / np.deg2rad(sensor.beamwidth_alongtrack) ** 2
        self.gamma_y = 8 * LOG2 / np.deg2rad(sensor.beamwidth_acrosstrack) ** 2

        self.L_gamma = sensor.alpha * sensor.altitude / (2 * self.gamma_y)

        # R15 Eq 37
        K = (
            (sensor.antenna_gain * sensor.wavelength * sensor.ndoppler) ** 2
            * self.Lx
            * self.Ly
            / (4 * np.pi * sensor.altitude**4)
            * np.sqrt(2 * np.pi)
            * Ag**2
            * sigma_g**2
        )  # unit: m^2 * m * m / m^4 =no unit
        # The sqrt(pulse_bandiwth) comes for sqrt(gl) in R15
        # The sqrt(2) is coming for R15 Eq 32,  the 1/sqrt(2) a,nd equation of Gamma_e which is twice G. This yields: 2/sqrt(2)
        self.Pu = K / np.sqrt(sensor.pulse_bandwidth) * np.sqrt(2)  # unit: s

        self.tau = delay_sampling_vector(sensor, oversampling_time)
        self.fdoppler = doppler_frequency_vector(sensor, oversampling_doppler)

    def delay_doppler_map(self, terrain_info: TerrainInfo):
        return self.delay_doppler_map_with_GO(terrain_info=terrain_info, mean_square_slope=np.inf)

    def delay_doppler_map_with_GO(self, terrain_info: TerrainInfo, mean_square_slope: npt.ArrayLike = np.inf):
        """Compute the delay doppler map for a surface"""
        if terrain_info.distribution != "normal":
            raise SMRTError("Dinardo18 is only available for a normally-distributed surface")

        sensor = self.sensor

        # tau = k / pulse_bandwidth
        tau = self.tau[:, np.newaxis]

        # look_angle = (Lx / sensor.altitude) * l  D18 Eq 16
        look_angle = sensor.wavelength / (2 * sensor.velocity) * self.fdoppler[np.newaxis, :]  # Raney 1998 Eq 6

        mean_square_slope = np.asarray(mean_square_slope)
        if mean_square_slope.ndim > 0:
            assert mean_square_slope.ndim == 1, f"shape error {mean_square_slope.ndim}"
            mean_square_slope = mean_square_slope[:, np.newaxis, np.newaxis]
            tau = tau[np.newaxis, ...]
            look_angle = look_angle[np.newaxis, ...]

        # D18 Eq 19  # the term with look_angle explains the X shape of the DDM (widening of the DDM for large fdoppler).

        sigma_c = np.sqrt(
            self.sigma_p**2 * (1 + (look_angle / self.theta_lim) ** 2) + (2 * terrain_info.sigma_surface / C_SPEED) ** 2
        )

        # visibly not present in the other models.

        sigma_s = terrain_info.sigma_surface / self.Lz

        # Eq 21
        nu = 1 / mean_square_slope  # decreasing of the backscatter as a function of the angle

        # Eq 25
        f = np.maximum(C_SPEED / (sensor.alpha * sensor.altitude) * tau, 0)

        Gamma_kl = np.exp(
            -self.gamma_y * sensor.roll_angle**2
            - nu * look_angle**2
            - self.gamma_x * (look_angle - sensor.pitch_angle) ** 2
            - (self.gamma_y + nu) * f
        ) * np.cosh(2 * self.gamma_y * sensor.roll_angle * np.sqrt(f))

        # gl = 1 / np.sqrt(sigma_g**2 + (2 * sigma_g * l * Lx**2 / Ly**2)**2 + sigma_s**2)  # R15 Eq 39
        # gl = tau / sigma_c / k
        gl = 1 / sensor.pulse_bandwidth / sigma_c  # deduced form D18 Eq 19 with tau = k / pulse_bandwith

        # D18 Eq 26 (formal, but devide by 0)
        # Tk = (1 + nu / gamma_y) - theta_roll / np.sqrt(f) * np.tanh(2 * gamma_y * theta_roll * np.sqrt(f))
        # D18 Eq 26 (valid)
        Tk = 1 + nu / self.gamma_y

        if sensor.roll_angle > 0:
            a = 2 * self.gamma_y * sensor.roll_angle
            sqrtf = np.sqrt(f)
            np.divide(
                np.tanh(a * sqrtf), sqrtf, out=np.full_like(sqrtf, a), where=sqrtf > 1e-10
            )  # this return a if sqrtf is small
            Tk -= sensor.roll_angle * a

        # D18 Eq 22
        return (
            self.Pu
            * Gamma_kl
            / np.sqrt(sigma_c)
            * (f0(tau / sigma_c) + terrain_info.sigma_surface / self.L_gamma * Tk * gl * sigma_s * f1(tau / sigma_c))
        )


def f0(xi: npt.NDArray):
    # D18 Eq 9
    x = xi**2 / 4  # to avoid the NaN in ive with neg order
    y = np.pi / 4 * np.sqrt(np.abs(xi)) * (scipy.special.ive(-0.25, x) + np.sign(xi) * scipy.special.ive(0.25, x))
    y[x < 1e-20] = np.pi * 2 ** (3 / 4) / (4 * scipy.special.gamma(3 / 4))  # D18 Eq 11
    return y


def f1(xi: npt.NDArray):
    # D18 Eq 10
    x = xi**2 / 4  # to avoid the NaN in ive with neg order
    y = (
        np.pi
        / 8
        * np.abs(xi) ** 1.5
        * (
            (scipy.special.ive(0.25, x) - scipy.special.ive(-0.75, x))
            + np.sign(xi) * (scipy.special.ive(-0.25, x) - scipy.special.ive(0.75, x))
        )
    )
    y[x < 1e-20] = -(2 ** (3 / 4)) * scipy.special.gamma(3 / 4) / 4  # D18 Eq 12
    return y
