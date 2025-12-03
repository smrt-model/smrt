"""This module calculates Delay Doppler Maps (DDM) based on Wingham et al. 2004.

The slant range correction as described by Wingham et al. 2004 is not implemented here; instead we use the numerical
correction (in `.delay_compensation`) applied directly on the delay Doppler map.

References:
    - D. J. Wingham, L. Phalippou, C. Mavrocordatos and D. Wallis, "The mean echo and echo cross product from a
      beamforming interferometric altimeter and their application to elevation measurement," in IEEE Transactions on
      Geoscience and Remote Sensing, vol. 42, no. 10, pp. 2305-2323, Oct. 2004, doi: 10.1109/TGRS.2004.834352.
"""

from typing import Optional

import numba
import numpy as np
import scipy.signal
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable

from smrt.core.globalconstants import C_SPEED
from smrt.core.terrain import TerrainInfo

from .delay_doppler_utils import (
    check_low_ndoppler,
    delay_compensation,
    delay_sampling_vector,
    numerical_convolution,
    ptr_function,
    sinc2,
)

SQRTLOG2 = 0.8325546111576977  # np.sqrt(LOG2)


class Wingham04(object):
    """Build a delay Doppler map model Wingham04.

    Args:
        sensor: SAR altimetry sensor object.
        oversampling_time: oversampling factor in time dimension.
        oversampling_doppler: oversampling factor in Doppler dimension.
        ptr_time: shape of the point target response in time dimension.
        ptr_doppler: shape of the point target response in Doppler dimension.
        slant_range_correction: whether to apply the slant range correction (True=SAR mode, False=pseudo-LRM mode)
        delay_window_widening: factor to widen the delay window. The default (1) is relevant when comparing with
            observed waveforms.
    """

    _backscatter_capability = (
        "constant"  # Wingham04 model assumes a constant backscatter, independent of the incidence angle.
    )

    def __init__(
        self,
        sensor,
        oversampling_time: int = 4,
        oversampling_doppler: int = 4,
        ptr_tau: Optional[str] = None,
        ptr_fdoppler: Optional[str] = None,
        slant_range_correction: bool = True,
        delay_window_widening: int = 1,
    ):
        super().__init__()

        self.sensor = sensor
        check_low_ndoppler(sensor.ndoppler, ddm_model=self)
        self.oversampling_time = oversampling_time
        self.oversampling_doppler = oversampling_doppler
        self.slant_range_correction = slant_range_correction
        self.delay_window_widening = delay_window_widening

        beamwidth = (sensor.beamwidth_alongtrack + sensor.beamwidth_acrosstrack) / 2  # get a 'circular' antenna pattern

        # from Eq 4 in wingham04
        self.gamma_a = np.sin(np.deg2rad(beamwidth) / 2) / SQRTLOG2

        # The angle determines the angular width of the beam. See Eq 12. -> depend on the burst
        self.zeta_b = (
            sensor.wavelength * sensor.pulse_repetition_frequency / (2 * sensor.ndoppler * sensor.velocity) / 2
        )  # almost same as R15 Eq 14 with wavelength = C_SPEED / sensor.frequency. The only difference is the factor 2.

        D0 = sensor.ndoppler**2  ## to concialate with W18 eq 18, the D0 in W04 must be Nb^2
        # Eq 37   # alpha on other paper is called Kappa in this paper
        self.K = (
            sensor.wavelength**2
            * sensor.antenna_gain**2
            * D0
            * C_SPEED
            / (32 * np.pi**2 * sensor.altitude**3 * sensor.alpha)
        )  # unit: s^-1 * D0  see eq 22 in W04.

        # by default the PTRs are ft of sinc**2
        self.ptr_tau = ptr_function(ptr_tau) if ptr_tau is not None else sinc2
        # self.ptr_fdoppler = ptr_function(ptr_fdoppler) if ptr_fdoppler is not None else sinc2

        self.tau = delay_sampling_vector(sensor, oversampling_time, self.delay_window_widening)

    def delay_doppler_map(self, terrain_info: TerrainInfo):
        """Compute the delay Doppler map for a surface."""
        sensor = self.sensor

        # look angle  W18 Eq 6

        L_2 = sensor.ndoppler // 2 * self.oversampling_doppler
        k = np.arange(-L_2 + 0.5, L_2 + 0.5) / self.oversampling_doppler
        xi_b = (
            k * np.pi * sensor.pulse_repetition_frequency / (sensor.ndoppler * sensor.wavenumber * sensor.velocity)
        )  # same as k * zeta_b * 2

        um = sensor.altitude / sensor.alpha * np.tan(np.deg2rad(terrain_info.slope_angle))
        args = (um, sensor.altitude, sensor.alpha, self.gamma_a, self.zeta_b, terrain_info.slope_direction)

        # In principle an integration over tau within each gate is required.
        # In the case of Wingham model. It is very important to compute the integral at the middle of the time gate
        # the reason is because the very tau=0 has a very strong return although it counts very little in terms of
        # surface area

        half_gate = 0.5 / sensor.pulse_bandwidth / self.oversampling_time

        # W04 Eq 22
        integral = [
            [
                scipy.integrate.quad(fn_integrand, 0, 2 * np.pi, args=(tau_ + half_gate, xi_b_) + args, epsabs=0)[0]
                for xi_b_ in xi_b
            ]
            for tau_ in self.tau
        ]

        # W04 Eq 22
        chi = (
            self.K
            # * sigma0
            * np.exp(
                -(2 / self.gamma_a**2)
                * (um**2 / sensor.altitude**2 + C_SPEED * self.tau[:, np.newaxis] / (sensor.altitude * sensor.alpha))
            )
            * np.array(integral)
        )  # unit: s^-1 * D0

        # compute the convolution numerically with ptr_f and ptr_t
        # the coef includes:
        # - coef in W04 Eq 17 for the factor c**2/4   # unit m^2 s^-2
        # - s(ct/2) has a factor 1/sigma_roughness which needs to be converted in time because in numerica_convolution,
        #   the pdf is in time. coef: 1 / (C_SPEED / 2)  # unit m^-1 s
        # - v(ct/2) has a factor 1 / (2 * np.pi C_SPEED)   # unit m^-1 s

        coef = (C_SPEED**2 / 4) / (C_SPEED / 2) / (2 * np.pi * C_SPEED)  # unit: no

        # # the convolution by v brings seconds and is not taken into account in numerical_convolution.
        # # we must add them here, with the coef delta_t
        # dtau = 1 / (sensor.pulse_bandwidth * self.oversampling_time)
        # coef *= dtau  # unit: s

        # compute the convolution numerically with ptr_t only
        ddm = coef * numerical_convolution(
            chi,  # unit: s^-1
            self.ptr_tau,
            None,  # the numerical convolution brings unit: s
            sensor,
            terrain_info,
            self.oversampling_time,
            self.oversampling_doppler,
        )  # unit: no

        if self.slant_range_correction:
            ddm = delay_compensation(sensor, ddm, self.oversampling_time, self.oversampling_doppler)

        return ddm[: sensor.ngate * self.oversampling_time, :]  # unit: no

    def impulse_map(
        self,
        grid_extent,
        terrain_info: TerrainInfo,
        gate=None,
        doppler_bin=None,
    ):
        """Compute the map of the impulse function for a given gate and doppler_bin.

        Args:
            grid_extent: size of the grid to return (in meters).
            terrain_info: terrain information used by the model.
            gate: gate number (0 corresponds to nadir; `nominal_gate` is not taken into account).
            doppler_bin: Doppler bin index.

        This map indicates the area "seen" for this gate and doppler_bin.
        """
        half_nx = half_ny = 500
        y, x = np.ogrid[-half_ny:half_ny, -half_nx:half_nx]
        x = x * grid_extent / (2 * half_nx)
        y = y * grid_extent / (2 * half_ny)

        sensor = self.sensor

        k = doppler_bin
        xi_b = (
            k * np.pi * sensor.pulse_repetition_frequency / (sensor.ndoppler * sensor.wavenumber * sensor.velocity)
        )  # same as k * zeta_b * 2

        um = sensor.altitude / sensor.alpha * np.tan(np.deg2rad(terrain_info.slope_angle))
        args = (um, sensor.altitude, sensor.alpha, self.gamma_a, self.zeta_b, terrain_info.slope_direction)

        vartheta = np.arctan2(y, x)
        rho2 = x**2 + y**2
        tau = self.sensor.alpha / (C_SPEED * sensor.altitude) * rho2  # eq 20 in W04

        delay_gate = tau * sensor.pulse_bandwidth
        mask = (delay_gate >= gate) & (delay_gate < gate + 1)

        return fn_integrand_njit(vartheta, tau, xi_b, *args) * mask


# def slant_correction(tau, xi_b, sensor, slope_angle, slope_direction):
#     """calculate slant correction using Wingham et al. 2004 equations that takes into account the terrain slope and latitude"""
#     lambda_s = 0

#     # Eq 35
#     tau_sl = sensor.alpha * sensor.altitude / C_SPEED * np.sin(xi_b) ** 2
#     # Eq 34
#     if slope_angle == 0:
#         rms = 0
#     else:
#         rms = (
#             sensor.altitude
#             - EARTH_RADIUS * np.sin(theta_s) * np.tan(slope_angle) * np.cos(lambda_s - slope_direction)
#             - sensor.altitude / (2 * sensor.alpha) * np.tan(slope_angle) ** 2
#         )
#     # Eq 37
#     tau_corrected = (
#         tau
#         + 2 * rms / C_SPEED
#         + tau_sl
#         - (sensor.altitude / (C_SPEED * sensor.alpha) * np.tan(slope_angle) ** 2 * np.sin(slope_direction) ** 2)
#     )
#     return tau_corrected, xi_b


@numba.jit(nopython=True, cache=True)
def fn_integrand_njit(vartheta, tau, xi_b, um, altitude, alpha, gamma_a, zeta_b, slope_direction):
    # W04 Eq 22
    # if tau < 0:
    #    return 0.0

    tau_factor = np.sqrt(tau * C_SPEED / (altitude * alpha))
    xi_mb = np.sin(xi_b) - um / altitude * np.cos(slope_direction)

    return np.exp(
        -4 * um / (gamma_a**2 * altitude) * tau_factor * np.cos(vartheta - slope_direction)
        - ((tau_factor * np.cos(vartheta) - xi_mb) / zeta_b)** 2
    )  # fmt: skip  # we simplified the two exponentials in one
    # return (
    #     np.exp(-4 * um / (gamma_a**2 * altitude) * tau_factor * np.cos(vartheta - slope_direction))
    #     * np.exp(- (tau_factor * np.cos(vartheta) - xi_mb)**2 / zeta_b**2)
    # )  # fmt: skip


# create a cfunc with numba
@numba.cfunc(float64(intc, CPointer(float64)))
def fn_integrand(size, xx):
    assert size == 9

    vartheta, tau, xi_b, um, altitude, alpha, gamma_a, zeta_b, slope_direction = numba.carray(xx, size)
    # W04 Eq 22
    if tau < 0:
        return 0.0

    tau_factor = np.sqrt(tau * C_SPEED / (altitude * alpha))
    xi_mb = np.sin(xi_b) - um / altitude * np.cos(slope_direction)

    return np.exp(
        -4 * um / (gamma_a**2 * altitude) * tau_factor * np.cos(vartheta - slope_direction)
        - ((tau_factor * np.cos(vartheta) - xi_mb) / zeta_b) ** 2
    )


fn_integrand = LowLevelCallable(fn_integrand.ctypes)
