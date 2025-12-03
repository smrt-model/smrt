"""This module calculates Delay Doppler Map (DDM) based on Halimi et al. 2014.

This DDM model is a semi-analytical method where the FSIR is calculated with an integral and the PTRf and PTRt
convolution are performed using numerical convolution.

The antenna footprint is circular and mispointing is not an option.

References:
    - Halimi, A., Mailhes, C., Tourneret, J.-Y., Thibaut, P., & Boy, F. (2014). A Semi-Analytical Model for
      Delay/Doppler Altimetry and Its Estimation Algorithm. IEEE Transactions on Geoscience and Remote Sensing, 52(7),
      4248–4258. https://doi.org/10.1109/tgrs.2013.2280595

"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from smrt.core.error import smrt_warn
from smrt.core.globalconstants import C_SPEED, LOG2
from smrt.core.terrain import TerrainInfo

from .delay_doppler_utils import (
    check_low_ndoppler,
    delay_compensation,
    delay_sampling_vector,
    doppler_frequency_vector,
    numerical_convolution,
    ptr_function,
    sinc2,
)


class Halimi14(object):
    """Build a delay Doppler map model Halimi14.

    Args:
        sensor: SAR altimetry sensor object.
        oversampling_time: oversampling factor in time dimension.
        oversampling_doppler: oversampling factor in Doppler dimension.
        ptr_time: shape of the point target response in time dimension.
        ptr_doppler: shape of the point target response in Doppler dimension.
        slant_range_correction: whether to apply the slant range correction (True=SAR mode, False=pseudo-LRM mode).
        delay_window_widening: factor to widen the delay window. The default (1) is relevant when comparing with
            observed waveforms.

    """

    def __init__(
        self,
        sensor,
        oversampling_time: int = 4,
        oversampling_doppler: int = 64,
        ptr_time: Optional[str] = None,
        ptr_doppler: Optional[str] = None,
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
        # gamma is after eq 3 in H14. This equation is slightly incorrect as the division by 2 is out of the sin...
        # it should be inside. This approx is valid for small angles only.
        self.gamma = 1 / (2 * LOG2) * np.sin(np.deg2rad(beamwidth)) ** 2

        # by default the PTRs are ft of sinc**2
        self.ptr_time = ptr_function(ptr_time) if ptr_time is not None else sinc2
        self.ptr_doppler = ptr_function(ptr_doppler) if ptr_doppler is not None else sinc2

        self.tau = delay_sampling_vector(sensor, oversampling_time, self.delay_window_widening)
        self.fdoppler = doppler_frequency_vector(sensor, oversampling_doppler)

    def delay_doppler_map(self, terrain_info: TerrainInfo):
        """Compute the delay Doppler map for a surface."""
        # the model do not accept terrain tilt and satellite misspointing
        if (terrain_info.slope_angle != 0) or (terrain_info.slope_direction != 0):
            smrt_warn("The Halimi14 model does not accept terrain slopes")

        sensor = self.sensor

        # Nf = len(fdoppler) // sensor.ndoppler
        # Nt = len(tau) // sensor.ngate

        tau = self.tau[:, np.newaxis]

        # fn = np.linspace(-sensor.ndoppler // 2, +sensor.ndoppler // 2, sensor.ndoppler * Nf) / sensor.burst_duration
        # fn = np.arange(-sensor.ndoppler // 2 * Nf + 0.5, +sensor.ndoppler // 2 * Nf) / Nf / sensor.burst_duration
        fn = self.fdoppler[np.newaxis, :]

        yn = sensor.altitude * sensor.wavelength / (2 * sensor.velocity) * fn  # Eq 11 H14

        rho2 = sensor.altitude * tau / sensor.alpha * C_SPEED  # in the text between Eq 3 and Eq 4 H14

        # compute yn / sqrt(rho2 - yn**2) but avoid warning for invalid values # Eq 12 H14
        # phi_n = np.arctan(yn / sqrtxn)  # Eq 12 H14
        xn = rho2 - yn**2
        valid = xn > 0
        tangente = np.full_like(xn, np.nan)
        np.divide(yn, np.sqrt(xn, where=valid), where=valid, out=tangente)
        phi_n = np.arctan(tangente)

        dphi_n = diff_cyclic(phi_n)

        Pu = (
            sensor.wavelength**2 * sensor.antenna_gain**2 * C_SPEED / (4 * (4 * np.pi) ** 2 * sensor.altitude**3)
        )  # in the text after Eq 3 H14. # unit: s^-1

        FSIR = (
            Pu
            * sensor.ndoppler**2
            / np.pi
            * np.exp(-4 * C_SPEED * tau / (sensor.alpha * self.gamma * sensor.altitude))
            * dphi_n
        )  # Eq 14 + the normalization with Nb^2 to match Wingham18/Ray17 convention  # unit: s^-1

        FSIR = np.nan_to_num(FSIR, copy=False, nan=0)  # unit: s^-1

        # compute the convolution numerically with ptr_f and ptr_t

        coef = (
            sensor.ndoppler * self.oversampling_doppler / sensor.pulse_repetition_frequency
        )  # convert n to doppler frequency # unit: s

        ddm = coef * numerical_convolution(
            FSIR,
            self.ptr_time,
            self.ptr_doppler,
            sensor,
            terrain_info,
            self.oversampling_time,
            self.oversampling_doppler,
        )  # unit: no

        if self.slant_range_correction:
            ddm = delay_compensation(sensor, ddm, self.oversampling_time, self.oversampling_doppler)

        return ddm[: sensor.ngate * self.oversampling_time, :]  # unit: no


def diff_cyclic(x: npt.NDArray) -> npt.NDArray:
    out = np.empty_like(x)
    out[:, :-1] = np.diff(x, axis=-1)
    out[:, -1] = x[:, 0] - x[:, 1]
    return out
