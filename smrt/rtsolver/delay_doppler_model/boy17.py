"""This module calculates Delay Doppler Maps (DDM) following ideas in Boy et al. 2017, with some interpretation as the
equations are not all explicited in the original paper.

The base idea of this model is to compute the FSSR as directly as possible from a x,y grid with 5 m resolution
encompassing the whole footprint.

The antenna footprint expression in x,y coordinates is taken from Buchhaupt et al.2018. The footprint center is not at
the center of the grid if the satellite is mis-pointing.

References:
    - Boy, F., Desjonqueres, J.-D., Picot, N., Moreau, T., & Raynal, M. (2017). CryoSat-2 SAR-Mode Over Oceans:
      Processing Methods, Global Assessment, and Benefits. IEEE Transactions on Geoscience and Remote Sensing, 55(1),
      148–158. https://doi.org/10.1109/tgrs.2016.2601958

"""

from typing import Optional

import numba
import numpy as np
import numpy.typing as npt

from smrt.core.globalconstants import C_SPEED, LOG4
from smrt.core.terrain import TerrainInfo, generate_dem

from .delay_doppler_utils import check_low_ndoppler, delay_compensation, numerical_convolution, ptr_function, sinc2


class Boy17(object):
    """Build a Boy17 Delay Doppler Model.

    Args:
        sensor: SAR altimetry sensor object.
        oversampling_time: oversampling factor in time dimension.
        oversampling_doppler: oversampling factor in doppler dimension.
        ptr_time: shape of the point target response in time dimension.
        ptr_doppler: shape of the point target response in doppler dimension.
        slant_range_correction: whether to apply the slant range correction (True=SAR mode, False=pseudo-LRM mode)
        delay_window_widening: factor to widen the delay window. The default (1) is relevant when comparing with
            observed waveforms.
        grid_space: spatial resolution of the x,y grid (in meters).
        with_earth_curvature: whether to consider earth curvature.

    """

    def __init__(
        self,
        sensor,
        oversampling_time: int = 4,
        oversampling_doppler: int = 4,
        ptr_time: Optional[str] = None,
        ptr_doppler: Optional[str] = None,
        slant_range_correction: bool = True,
        delay_window_widening: int = 1,
        grid_space: float = 5,
        with_earth_curvature: bool = True,
    ):
        super().__init__()

        self.sensor = sensor
        check_low_ndoppler(sensor.ndoppler, ddm_model=self)
        self.oversampling_time = oversampling_time
        self.oversampling_doppler = oversampling_doppler
        self.slant_range_correction = slant_range_correction
        self.with_earth_curvature = with_earth_curvature
        self.delay_window_widening = delay_window_widening

        # grid space
        self.grid_space = grid_space  # 10 m should be the default
        max_drang = (sensor.ngate - sensor.nominal_gate) / (
            sensor.pulse_bandwidth * (2 / C_SPEED)
        )  # max range (-altitude)
        if self.with_earth_curvature:
            max_drang /= sensor.alpha

        max_x = 2 * np.sqrt(
            (max_drang + sensor.altitude) ** 2 - sensor.altitude**2
        )  # max horizontal extent of the footprint
        self.half_nx = self.half_ny = int(max_x / self.grid_space)

        self.gamma_y = np.sin(np.deg2rad(sensor.beamwidth_acrosstrack)) ** 2 / LOG4
        self.gamma_x = np.sin(np.deg2rad(sensor.beamwidth_alongtrack)) ** 2 / LOG4

        # Doppler cell resolution (after Eq 1 in B17)
        self.Lx = sensor.altitude * sensor.wavelength / (2 * sensor.velocity * sensor.burst_duration)

        # by default the PTRs are ft of sinc**2
        self.ptr_time = ptr_function(ptr_time) if ptr_time is not None else sinc2
        self.ptr_doppler = ptr_function(ptr_doppler) if ptr_doppler is not None else sinc2

    def delay_doppler_map(self, terrain_info: TerrainInfo):
        return self.delay_doppler_map_with_sigma0func(terrain_info=terrain_info, sigma0=1)

    def delay_doppler_map_with_sigma0func(self, terrain_info: TerrainInfo, sigma0: npt.ArrayLike = 1):
        """Compute the delay doppler map for a surface"""
        sensor = self.sensor

        # nx_2 = ny_2 = 2500  # e.g. 2 * nx_2 gridpoint = 5000 * 5 = 25 km !

        # when the satellite is at nadir, it would be possible to divide by 2 the y-axis (due to symmetry)
        y, x = np.ogrid[-self.half_ny : self.half_ny, -self.half_nx : self.half_nx]
        x *= self.grid_space  # (B17 recommends 5 m)
        y *= self.grid_space  # (B17 recommends 5 m)

        dem = generate_dem(terrain_info, x, y)

        Nw = self.delay_window_widening  # window widening

        # calculate the range (distance from the satellite to the point x,y)
        dz = sensor.altitude - dem
        rho2 = x**2 + y**2

        alpha = sensor.alpha if self.with_earth_curvature else 1

        rang = alpha * (np.sqrt(dz**2 + rho2) - sensor.altitude)
        # deduce the gate range.
        irange = (
            np.floor(rang * (2 / C_SPEED * sensor.pulse_bandwidth * self.oversampling_time)).astype(int)
            + sensor.nominal_gate * self.oversampling_time
        )

        # compute the doppler frequency
        idoppler = (
            np.floor(x * (self.oversampling_doppler / self.Lx)).astype(int)
            + sensor.ndoppler * self.oversampling_doppler // 2
        )

        # after Equation 3 in Halimi et al. 2014
        Pu = (
            sensor.wavelength**2 * sensor.antenna_gain**2 * C_SPEED / (4 * (4 * np.pi) ** 2 * sensor.altitude**3)
        )  # in the text after Eq 3 H14. # unit: s^-1

        # grid_space**2 is due to the integration over x and y
        FSSR0 = (Pu * sensor.ndoppler**2 * self.grid_space**2) * antenna_gain_cartesian_coordinates(
            x, y, sensor.altitude, self.gamma_x, self.gamma_y, sensor.pitch_angle, sensor.roll_angle
        ) ** 2  # the normalization with Nb^2 to match Wingham18/Ray17 convention   # unit: m^2 s^-1

        # transform sigma0 in a list for the vectorization
        try:
            sigma0_list = list(sigma0)  # type: ignore
            squeeze_first_dim = False
        except TypeError:
            sigma0_list = [sigma0]
            squeeze_first_dim = True

        # reserve the memory for the DDMs
        ddm = np.empty(
            (len(sigma0_list), sensor.ngate * self.oversampling_time * Nw, sensor.ndoppler * self.oversampling_doppler)
        )

        # loop over sigma0_list
        for j, sigma0 in enumerate(sigma0_list):
            if callable(sigma0):
                # global incidence angle at the x, y position. Neglecting the local slope.
                # theta_i = np.arctan(np.sqrt(rho2) / dz)
                # cos(theta_i) = dz / (np.sqrt(rho2 + dz**2))
                mu_i = np.minimum(dz / (rang + sensor.altitude), 1.0)
                sigma0_values = sigma0(mu_i)
            else:
                sigma0_values = sigma0

            FSSR = sigma0_values * FSSR0

            ddm[j] = compute_delay_doppler_map(
                irange,
                idoppler,
                self.oversampling_doppler // 2,
                FSSR,
                ndoppler=sensor.ndoppler * self.oversampling_doppler,
                ngate=sensor.ngate * self.oversampling_time * Nw,
            )  # unit: m^2 s^-1

            # 2D convolution with the PTRt and PTRf
            # compute the convolution numerically with ptr_t only

            ddm[j] = numerical_convolution(
                ddm[j], self.ptr_time, None, sensor, None, self.oversampling_time, self.oversampling_doppler
            )  # unit: m^2

            if self.slant_range_correction:
                ddm[j] = delay_compensation(
                    sensor, ddm[j], self.oversampling_time, self.oversampling_doppler, self.with_earth_curvature
                )

        if squeeze_first_dim:
            ddm = ddm[0]
        return ddm[..., 0 : sensor.ngate * self.oversampling_time, :]

    def impulse_map(self, gate, doppler_bin, grid_extent, terrain_info: TerrainInfo):
        """Compute the map of the impulse function for a given gate and doppler_bin.

        This map indicates the area "seen" for this gate and doppler_bin.

        Args:
            gate: gate number (0 correspond to the nadir, the nominal_gate is not taken into account).
            doppler_bin: doppler bin.
            grid_extent: size of the grid to return (in meter).
        """
        half_nx = half_ny = 500
        y, x = np.ogrid[-half_ny:half_ny, -half_nx:half_nx]
        x = x * grid_extent / (2 * half_nx)
        y = y * grid_extent / (2 * half_ny)

        dem = 0  # generate_dem(terrain_info, x, y)

        # calculate the range (distance from the satellite to the point x,y)
        dz = self.sensor.altitude - dem
        rho2 = x**2 + y**2

        alpha = self.sensor.alpha if self.with_earth_curvature else 1

        rang = alpha * (np.sqrt(dz**2 + rho2) - self.sensor.altitude)

        # deduce the gate range.
        irange = np.floor(rang * (2 / C_SPEED * self.sensor.pulse_bandwidth))

        # compute the doppler frequency
        idoppler = np.floor(x * (1 / self.Lx))

        return ((irange == gate) & (idoppler == doppler_bin)).astype(np.float64)
        # return ((irange == gate)).astype(np.float64)


@numba.jit(nopython=True, cache=True)
def compute_delay_doppler_map(
    irange: npt.NDArray[np.int_],
    idoppler: npt.NDArray[np.int_],
    idoppler_halfwidth: int,
    FSSR: npt.NDArray[np.floating],
    ndoppler: int,
    ngate: int,
) -> npt.NDArray[np.floating]:
    ddm = np.zeros((ngate, ndoppler))

    for iy in range(irange.shape[0]):
        for ix in range(irange.shape[1]):
            rn = irange[iy, ix]
            if 0 <= rn < ddm.shape[0]:
                fn = idoppler[0, iy]
                if 0 <= fn < ddm.shape[1]:
                    ddm[rn, fn - idoppler_halfwidth : fn + idoppler_halfwidth + 1] += FSSR[iy, ix]
    return ddm


@numba.jit
def antenna_gain_cartesian_coordinates(
    x: npt.NDArray,
    y: npt.NDArray,
    altitude: npt.NDArray,
    gamma_x: npt.NDArray,
    gamma_y: npt.NDArray,
    pitch_angle: npt.NDArray,
    roll_angle: npt.NDArray,
) -> npt.NDArray[np.floating]:
    # Antenna gain in the x, y plan assuming a gaussian shape with roll and pitch angle
    # given by Buchhaupt et al.2018

    # Eq 24 in Buchhaupt et al. 2018

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

    mu = (gamma_y - gamma_x) / gamma_x  # asymmetry parameter of the antenna gain
    rho2 = x**2 + y**2

    fxx = (1 - cosxi2 * tanxi_p**2) / (altitude**2 + rho2) + mu * (cosxi + cosxi2) / (
        2 * altitude * np.sqrt(altitude**2 + rho2)
    )
    fyy = (1 - cosxi2 * tanxi_r**2) / (altitude**2 + rho2)

    fx = 2 * altitude * cosxi2 * tanxi_p / (altitude**2 + rho2) + mu * tanxi_p * (cosxi + cosxi2) / np.sqrt(
        altitude**2 + rho2
    )
    fy = 2 * altitude * cosxi2 * tanxi_r / (altitude**2 + rho2)

    f00 = altitude**2 * sinxi2 / (altitude**2 + rho2) + mu * altitude * tanxi_p**2 * (cosxi + cosxi2) / (
        2 * np.sqrt(altitude**2 + rho2)
    )

    fxy = 2 * cosxi2 * tanxi_p * tanxi_r / (altitude**2 + rho2)
    return np.exp(-2 / gamma_y * (f00 + x**2 * fxx + y**2 * fyy - x * fx - y * fy - x * y * fxy))
