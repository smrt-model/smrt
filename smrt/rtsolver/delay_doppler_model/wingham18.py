"""This module calculates Delay Doppler Maps (DDM) based on Wingham et al. 2018.

References:
    - D. J. Wingham, K. A. Giles, N. Galin, R. Cullen, T. W. K. Armitage and W. H. F. Smith, "A Semianalytical Model of
      the Synthetic Aperture, Interferometric Radar Altimeter Mean Echo, and Echo Cross-Product and Its
      Statistical Fluctuations," in IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 5, pp.
      2539-2553, May 18, doi: 10.1109/TGRS.2017.2756854.

"""

from functools import lru_cache
from typing import Optional

import numba
import numpy as np
import scipy.signal
from numba.extending import is_jitted
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable

from smrt.core.error import SMRTError
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


class Wingham18(object):
    """Build a delay Doppler map model Wingham18.

    Args:
        sensor: SAR altimetric sensor
         oversampling_time: oversampling factor in time dimension.
        oversampling_doppler: oversampling factor in doppler dimension.
        slant_range_correction: whether to apply slant_range or not, and if yes how. This module can performed
            "analytical" slant range correction applying the correction in the time delay calculation. In contrast
            "numerical" is closer to SAR processing and to real observations, the DDM is first calculated without
            correction, and migration is then applied on the DDM. If set to True, "numerical" is used if the widening
            factor is <= 1, otherwise "analytical" is used.
        ptr_time: shape of the PTR_time function.
        ptr_doppler:  shape of the PTR_doppler function.
        delay_window_widening: factor controlling the number of gates used to compute the DDM before the slant range
            correction. Defaults to 1. See the `slant_range_correction` attribute.
    """

    _backscatter_capability = (
        "constant"  # Wingham18 model assumes a constant backscatter, independent of the incidence angle.
    )

    def __init__(
        self,
        sensor,
        oversampling_time: int = 4,
        oversampling_doppler: int = 4,
        slant_range_correction: bool | str = True,
        ptr_time: Optional[str] = None,
        ptr_doppler: Optional[str] = None,
        delay_window_widening: int = 1,
    ):
        super().__init__()

        self.sensor = sensor
        self.oversampling_time = oversampling_time
        self.oversampling_doppler = oversampling_doppler
        self.delay_window_widening = delay_window_widening

        if slant_range_correction is True:
            if delay_window_widening <= 1:
                self.slant_range_correction = "numerical"
            else:
                self.slant_range_correction = "analytical"
        else:
            self.slant_range_correction = slant_range_correction

        if self.slant_range_correction not in [False, "analytical", "numerical"]:
            raise SMRTError("slant_range_correction attribute must be False, True, 'analytical' or 'numerical'")

        # gamma is after W18 eq 5
        self.gamma_1 = np.deg2rad(sensor.beamwidth_alongtrack / 2) / SQRTLOG2
        self.gamma_2 = np.deg2rad(sensor.beamwidth_acrosstrack / 2) / SQRTLOG2

        self.xi_b0 = np.pi / (sensor.ndoppler * sensor.wavenumber * sensor.velocity / sensor.pulse_repetition_frequency)

        D0 = 1  ## based on W18 eq 18
        # coef in Eq 16 (see also Eq 37 in W04)   # alpha on other paper is called Kappa in this paper
        self.K = (
            sensor.wavelength**2
            * sensor.antenna_gain**2
            * D0
            * C_SPEED
            / (32 * np.pi**2 * sensor.altitude**3 * sensor.alpha)
        )  # unit: s^-1

        # by default the PTRs are ft of sinc**2
        self.ptr_time = ptr_function(ptr_time) if ptr_time is not None else sinc2

        self.ptr_doppler = ptr_doppler if ptr_doppler is not None else "sar_wingham18"
        if self.ptr_doppler == "sar_wingham04":
            check_low_ndoppler(sensor.ndoppler, ddm_model=self)

        self.tau = delay_sampling_vector(sensor, oversampling_time, self.delay_window_widening)

    def delay_doppler_map(self, terrain_info: TerrainInfo):
        """Compute the delay doppler map for a surface"""
        sensor = self.sensor

        mu = sensor.pitch_angle  # along-track pitch
        chi = sensor.roll_angle  # across-track roll
        beta = 0  # across-track surface slope  # TODO: to compute from terrain slope

        # Eq 9 interferometric angle
        theta = chi + beta / sensor.alpha

        def PFS(tau, xi_b):
            # Eq 8
            if self.slant_range_correction == "analytical":
                rho2 = C_SPEED * tau / (sensor.alpha * sensor.altitude) + xi_b**2
            else:
                rho2 = C_SPEED * tau / (sensor.alpha * sensor.altitude)

            if rho2 <= 0:  # take into account the Heaviside in 4.
                return 0.0
            rho = np.sqrt(rho2)

            j1_integrand = self.get_j_integrand(1)
            j1 = scipy.integrate.quad(j1_integrand, 0, 2 * np.pi, args=(rho, xi_b), epsabs=0)[0]

            if mu != 0:
                j2_integrand = self.get_j_integrand(2)
                j2 = scipy.integrate.quad(j2_integrand, 0, 2 * np.pi, args=(rho, xi_b), epsabs=0)[0]

                j3_integrand = self.get_j_integrand(3)
                j3 = scipy.integrate.quad(j3_integrand, 0, 2 * np.pi, args=(rho, xi_b), epsabs=0)[0]
            else:
                j2 = 0.0
                j3 = 0.0

            if theta != 0:
                j4_integrand = self.get_j_integrand(4)
                j4 = scipy.integrate.quad(j4_integrand, 0, 2 * np.pi, args=(rho, xi_b), epsabs=0)[0]
            else:
                j4 = 0

            # Eq 13
            return self.K * (
                +j1
                + mu * 4 * rho / self.gamma_1**2 * j2
                + mu**2 * (8 * rho2 / self.gamma_1**4 * j3 - 2 / self.gamma_1**2 * j1)
                + theta**2 * (8 * rho2 / self.gamma_2**4 * j4 - 2 / self.gamma_2**2 * j1)
            )

        PFS = np.vectorize(PFS)

        xi_b = np.arange(
            -sensor.ndoppler * self.oversampling_doppler // 2 + 0.5,
            sensor.ndoppler * self.oversampling_doppler // 2 + 0.5,
        )[np.newaxis, :] * (self.xi_b0 / self.oversampling_doppler)

        # In principle an integration over tau within each gate is required.
        # In the case of Wingham model. It is very important to compute the integral at the middle of the time gate
        # the reason is because the very tau=0 has a very strong return although it counts very little in terms of
        # surface area

        half_gate = 0.5 / sensor.pulse_bandwidth / self.oversampling_time

        chi = PFS(self.tau[:, np.newaxis] + half_gate, xi_b)  # unit: s^-1

        # compute the convolution numerically with ptr_t only
        ddm = numerical_convolution(
            chi, self.ptr_time, None, sensor, terrain_info, self.oversampling_time, self.oversampling_doppler
        )  # unit: no

        if self.slant_range_correction == "numerical":
            ddm = delay_compensation(sensor, ddm, self.oversampling_time, self.oversampling_doppler)

        return ddm[: sensor.ngate * self.oversampling_time, :]  # unit: no

    def get_j_integrand(self, j: int):
        """Return the integrand in j function defined in W18 Eq 14.

        For efficiency reason they are return as compiled C function (more precisely scipy.LowLevelCallable) suitable for
        the integration with scipy.integrate.quad. To avoid the cost of mulitple compilations, the result is cached. Changing the sensor, the kernel or the ptr_doppler
        """
        return make_j_integrand(
            self.sensor.wavelength,
            self.sensor.velocity,
            self.sensor.pulse_repetition_frequency,
            self.sensor.ndoppler,
            self.gamma_1,
            self.gamma_2,
            self.ptr_doppler,
            j,
        )


@lru_cache
def make_j_integrand(
    wavelength: float,
    velocity: float,
    pulse_repetition_frequency: float,
    ndoppler: int,
    gamma_1: float,
    gamma_2: float,
    ptr_doppler: str,
    j: int,
):
    def ptrf_sar_wingham18(phi: float):
        # valid for the SAR mode only according to W18
        # eq 18 in W18 without the pi which is an error ! See eq 14 in Landy 18
        k0 = 2 * np.pi / wavelength
        angle = (k0 * velocity / pulse_repetition_frequency) * phi
        return (np.sin(ndoppler * angle) / np.sin(angle)) ** 2 if np.abs(angle) > 1e-6 else ndoppler**2

    def ptrf_sarin_wingham18(phi: float):
        # valid for the SARIN mode
        n = np.arange(0, ndoppler)
        k0 = 2 * np.pi / wavelength
        # Eq 17 in W18  # this is different in the Cryosat Handbook
        s = np.sum(
            (0.54 - 0.46 * np.cos(2 * np.pi * n / (ndoppler - 1) - np.pi))
            * np.exp(-2j * (k0 * velocity / pulse_repetition_frequency * (n - (ndoppler - 1) / 2)) * phi)
        )
        return s.real**2 + s.imag**2

    def ptrf_sar_wingham04(phi: float):
        zeta_b = wavelength * pulse_repetition_frequency / (2 * ndoppler * velocity) / 2
        return ndoppler**2 * np.exp(-((phi / zeta_b) ** 2))

    # compile the kernel and the doppler gain function
    ptrf_map = {
        "sar_wingham18": ptrf_sar_wingham18,
        "sarin_wingham18": ptrf_sarin_wingham18,
        "sar_wingham04": ptrf_sar_wingham04,
    }

    ptr_doppler_function = numba.jit(ptrf_map[ptr_doppler], nopython=True)
    jitted_kernel = numba.jit(kernels[j], nopython=True) if not is_jitted(kernels[j]) else kernels[j]

    # create a cfunc with numba
    @numba.cfunc(float64(intc, CPointer(float64)))
    def jn_integrand(size, xx):
        assert size == 3
        vartheta, rho, xi = numba.carray(xx, size)  # unpack the arguments

        # Eq 14
        cos = np.cos(vartheta)
        cos2 = cos**2
        return (
            jitted_kernel(cos)
            * ptr_doppler_function(rho * cos - xi)
            * np.exp(-2 * rho**2 * (cos2 / gamma_1**2 + (1 - cos2) / gamma_2**2))
        )

    return LowLevelCallable(jn_integrand.ctypes)


@numba.jit(nopython=True, cache=True)
def kernel1(cos):
    return 1.0


@numba.jit(nopython=True, cache=True)
def kernel2(cos):
    return cos


@numba.jit(nopython=True, cache=True)
def kernel3(cos):
    return cos**2


@numba.jit(nopython=True, cache=True)
def kernel4(cos):
    return 1 - cos**2


kernels = {1: kernel1, 2: kernel2, 3: kernel3, 4: kernel4}
