"""This module calculates Delay Doppler Maps based on Buchhaupt et al. 2018.

It assumes that the PTRx and PTRdoppler are sinc**2. Can be easily changed if the PTRs have a known Fourier Transform

References:
    - Buchhaupt, C., Fenoglio-Marc, L., Dinardo, S., Scharroo, R., & Becker, M. (2018). A fast convolution based
      waveform model for conventional and unfocused SAR altimetry. Advances in Space Research, 62(6), 1445–1463.
      https://doi.org/10.1016/j.asr.2017.11.039
"""

from typing import Optional

# from smrt.core.optional_numba import numba
import numba  # numba is not optional
import numpy as np
import numpy.typing as npt
import scipy.integrate

from smrt.core.error import SMRTError, smrt_warn
from smrt.core.globalconstants import C_SPEED, LOG4
from smrt.core.terrain import TerrainInfo

from .delay_doppler_utils import delay_compensation, ft_ptr_function, triangular_function


class Buchhaupt18(object):
    """Build a Buchhaupt18 Delay Doppler Model.

    Args:
        sensor: SAR altimetry sensor object.
        oversampling_time: oversampling factor in time dimension.
        oversampling_doppler: oversampling factor in doppler dimension.
        ptr_time: shape of the point target response in time dimension.
        ptr_doppler: shape of the point target response in doppler dimension.
        slant_range_correction: whether to apply slant_range or not, and if yes how. This module can performed
            "analytical" slant range correction applying the correction in the time delay calculation. In contrast
            "numerical" is closer to SAR processing and to real observations, the DDM is first calculated without
            correction, and migration is then applied on the DDM. If set to True, "numerical" is used if the widening
            factor is <= 1, otherwise "analytical" is used.
        delay_window_widening: factor to widen the delay window. The default (1) is relevant when comparing with
            observed waveforms.

    """

    _backscatter_capability = "geometrical_optics"  # Buchhaupt18 model assumes the GO model for the rough surface

    def __init__(
        self,
        sensor,
        oversampling_time: int = 4,
        oversampling_doppler: int = 4,
        ptr_time: Optional[str] = None,
        ptr_doppler: Optional[str] = None,
        slant_range_correction: bool | str = True,
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

        if self.oversampling_doppler < 4:
            # recommended in Sec 3.2.1 (beginning of column b)
            smrt_warn("A minimum oversampling of 4 is recommended by Buchhaupt 2018 in Sec 3.2.1 ")

        self.burst_duration = 1 / sensor.pulse_repetition_frequency * sensor.ndoppler  # 3.5e-3
        self.Lx = (
            sensor.altitude * sensor.wavelength / (2 * sensor.velocity * self.burst_duration)
        )  # Doppler cell resolution (after Eq 1 in B18)  # unit: m

        # by default the PTRs are ft of sinc**2
        self.ft_ptr_time = ft_ptr_function(ptr_time) if ptr_time is not None else triangular_function
        self.ft_ptr_doppler = ft_ptr_function(ptr_doppler) if ptr_doppler is not None else triangular_function

        self.gamma_y = np.sin(np.deg2rad(sensor.beamwidth_acrosstrack)) ** 2 / LOG4
        gamma_x = np.sin(np.deg2rad(sensor.beamwidth_alongtrack)) ** 2 / LOG4
        self.mu = (self.gamma_y - gamma_x) / gamma_x  # asymmetry parameter of the antenna gain

    def delay_doppler_map(self, terrain_info: TerrainInfo):
        return self.delay_doppler_map_with_GO(terrain_info=terrain_info, mean_square_slope=np.inf)

    def delay_doppler_map_with_GO(self, terrain_info: TerrainInfo, mean_square_slope: npt.ArrayLike = np.inf):
        """Compute the delay doppler map for a surface."""
        sensor = self.sensor

        # convert the tau coordinate into the FT space
        Nt = 2**3  # # widening factor of the receiving window value based on discussion in sec 3.1.3
        N = sensor.ngate * self.oversampling_time * Nt
        n = np.hstack((np.arange(0, N // 2), np.arange(-N // 2, 0)))[
            :, np.newaxis
        ]  # frequencies organised for scipy.fft
        # fn = 1 / (tau[1] - tau[0]) / N * (n - 1)  # frequency in the time domain
        fn = (n - 1) * self.oversampling_time * sensor.pulse_bandwidth / N

        # sampling in the eta domain
        # after eq 51
        # convert the fdoppler into x coordinate and then into the FT space (=slow-time domain)
        L = sensor.ndoppler * self.oversampling_doppler
        m = np.hstack((np.arange(0, L // 2), np.arange(-L // 2, 0)))[
            np.newaxis, :
        ]  # frequencies organised for scipy.fft

        etam = (m / L) / (self.Lx / self.oversampling_doppler)
        # etam = 1 / (fdoppler[1] - fdoppler[0]) * sensor.pulse_repetition_frequency / sensor.ndoppler / self.Lx  * (m / L)

        mean_square_slope = np.asarray(mean_square_slope)
        if mean_square_slope.ndim > 0:
            assert mean_square_slope.ndim == 1, f"shape error {mean_square_slope.ndim}"
            mean_square_slope = mean_square_slope[:, np.newaxis, np.newaxis]
            etam = etam[np.newaxis, ...]
            fn = fn[np.newaxis, ...]

        # calculate the ft_prt of tau and fdoppler
        ft_ptr_time = self.ft_ptr_time(fn, self.sensor.pulse_bandwidth)  # B18 Eq 32  # unit: s
        ft_ptr_x = sensor.ndoppler**2 * self.ft_ptr_doppler(
            etam, 1 / self.Lx
        )  # B18 Eq 33  #Nb^2 is to conform with Wingham definition # unit: m

        # surface pdf
        assert terrain_info.distribution == "normal"
        sigma_s = 2 * terrain_info.sigma_surface / C_SPEED  # convert sigma_s from meter to second
        PDF = np.exp(-2 * (np.pi * sigma_s * fn) ** 2)  # B18 Eq 35  unit: no

        # geometrical optics
        alpha0 = 1 / (2 * mean_square_slope)

        # coef A after Eq 29. We separate this coef in a constant term (calculated here) and the variable term calculated in ft2_FSSR
        Lp = 1  # two-way atmospheric transmittance
        G0 = sensor.antenna_gain
        Aconst = (
            sensor.wavelength**2 * G0**2 * C_SPEED / (4 * (4 * np.pi) ** 2 * Lp * sensor.altitude**3 * sensor.alpha)
        )  # Hz

        # add the dx and dtau comming for the Fourier Transform, and add the conversion to fdoppler
        dx = self.Lx / self.oversampling_doppler  # unit: m
        dtau = 1 / (sensor.pulse_bandwidth * self.oversampling_time)  # unit: s
        Aconst_ft2norm = Aconst / (dx * dtau)  # unit: m^-1 s^-2

        p = ft2_FSSR(
            fn,
            etam,
            sensor.altitude,
            sensor.alpha,
            self.gamma_y,
            self.mu,
            alpha0,
            sensor.pitch_angle,
            sensor.roll_angle,
        )  # in s
        ft2_Pd = p * ft_ptr_x * (PDF * ft_ptr_time)  # unit: s * m * no * s = m s^2

        if self.slant_range_correction == "analytical":
            # xd = m * self.Lx / self.oversampling_doppler
            xd = etam * L * (self.Lx / self.oversampling_doppler) ** 2

            ft_Pd = scipy.fft.ifft(ft2_Pd, axis=-1)
            ft_Pd *= np.exp(
                2j
                * np.pi
                * fn
                * (2 / C_SPEED * np.sqrt((sensor.altitude**2 + sensor.alpha * xd**2)) - 2 * sensor.altitude / C_SPEED)
            )

            ddm = scipy.fft.ifft(ft_Pd, axis=-2).real * (Aconst_ft2norm)  # unit: no
        else:
            # perform the two FT in one scipy call
            ddm = scipy.fft.ifft2(ft2_Pd, axes=(-2, -1)).real * (Aconst_ft2norm)  # unit: no

        ddm = scipy.fft.fftshift(ddm, axes=(-2, -1))
        # remove the Nt lengthening
        # taushift = int(tau[0] * sensor.pulse_bandwidth * self.oversampling_time)
        taushift = -sensor.nominal_gate * self.oversampling_time

        ddm = ddm[..., N // 2 + taushift : N // 2 + taushift + sensor.ngate * self.oversampling_time, :]  # unit: no

        if self.slant_range_correction == "numerical":
            if len(ddm.shape) == 3:
                for j in range(ddm.shape[0]):
                    ddm[j, :, :] = delay_compensation(
                        sensor, ddm[j, :, :], self.oversampling_time, self.oversampling_doppler
                    )
            elif len(ddm.shape) == 2:
                ddm = delay_compensation(sensor, ddm, self.oversampling_time, self.oversampling_doppler)
            else:
                raise NotImplementedError("Unsupported shape for the DDM")

        return ddm


@numba.jit(nopython=True, cache=True)
def ft2_FSSR(
    f: npt.NDArray,
    eta: npt.ArrayLike,
    altitude: np.floating,
    alpha: np.floating,
    gamma_y: npt.NDArray,
    mu: npt.NDArray,
    alpha0: npt.ArrayLike,
    pitch_angle: npt.ArrayLike,
    roll_angle: npt.ArrayLike,
) -> npt.NDArray:
    # calculate the FSSR in the Fourier space for both parameters f<->tau and eta<->fdoppler

    xi_p = pitch_angle
    xi_r = roll_angle

    # xi = np.arctan(np.sqrt(np.tan(xi_p)**2 + np.tan(xi_r)**2))
    # cosxi = np.cos(xi)
    tanxi2 = np.tan(xi_p) ** 2 + np.tan(xi_r) ** 2
    cosxi = 1 / np.sqrt(1 + tanxi2)

    cosxi2 = cosxi**2
    sinxi2 = 1 - cosxi2
    tanxi_p = np.tan(xi_p)
    tanxi_r = np.tan(xi_r)

    # coef A after Eq 29. We separate this coef in a constant term (calculated outside this function) and the variable term
    # (depend on xi xi_p)
    # Aconst = wavelength**2 * G0**2 * sigma0 * C_SPEED / (4 * (4 * np.pi)**2 * Lp * altitude**3 * alpha)
    Avar = np.exp(-4 / gamma_y * (sinxi2 + mu / 2 * tanxi_p**2 * (cosxi + cosxi2)))

    coef = 4 / gamma_y

    one_way_1 = C_SPEED / (alpha * altitude)
    one_way_12 = np.sqrt(one_way_1)

    # B18 Eq 26
    s_x = (
        coef * one_way_1 * (1 - cosxi2 * tanxi_p**2 + mu / 2 * (cosxi + cosxi2)) + one_way_1 * alpha0 + 2j * np.pi * f
    )  # in Hz

    # B18 Eq 26
    beta_x = coef * one_way_12 * (cosxi2 * tanxi_p + mu / 3 * tanxi_p * (cosxi + cosxi2))

    # B18 Eq 26
    s_y = coef * one_way_1 * (1 - cosxi2 * tanxi_r**2) + one_way_1 * alpha0 + 2j * np.pi * f  # in Hz

    # B18 Eq 26
    beta_y = coef * one_way_12 * cosxi2 * tanxi_r

    # B18 Eq 26
    beta_xy = coef * one_way_1 * cosxi2 * tanxi_r * tanxi_p

    ft2_FSSR = (
        Avar
        / np.sqrt(s_x * s_y - beta_xy**2 / 4)  # unit: Hz
        * np.exp(
            beta_y**2 / s_y
            + (beta_x + np.pi * 1j * np.sqrt(C_SPEED * altitude / alpha) * eta + beta_y * beta_xy / s_y) ** 2
            / (s_x - beta_xy**2 / s_y)
        )
    )

    return ft2_FSSR  # unit: s
