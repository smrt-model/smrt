"""This module calculates Delay Doppler Map (DDM) based on Landy et al. 2019.

The orginal code for this model is: https://github.com/jclandy/FBEM/blob/master/Facet_Echo_Model.m

The main difference with respect to Landy et al. 2019 and their implementation is the use the sinc^2 transform
(Greenhard et al. 2006), to significantly accelerate the calculation.

Note that the footprint center is always at the center of the grid even if the satellite is mis-pointing (differ from
our Boy17 implementation).

References:
    - Landy, J. C., Tsamados, M., & Scharien, R. K. (2019). A Facet-Based Numerical Model for Simulating SAR Altimeter
      Echoes From Heterogeneous Sea Ice Surfaces. IEEE Transactions on Geoscience and Remote Sensing, 57(7), 4164-4180.
      https://doi.org/10.1109/tgrs.2018.

    - Greengard, L., Lee, J.-Y., & Inati, S. (2006). The fast sinc transform and image reconstruction from
      nonuniform samples ink-space. Communications in Applied Mathematics and Computational Science, 1(1), 121-131.
      https://doi.org/10.2140/camcos.2006.1.121

"""

from typing import Optional

import numba
import numpy as np
import numpy.typing as npt

from smrt.core.error import SMRTError
from smrt.core.globalconstants import C_SPEED
from smrt.core.terrain import TerrainInfo, generate_dem

from .delay_doppler_utils import delay_compensation, delay_sampling_vector, ptr_function, sinc2

try:
    import finufft
except ImportError:
    finufft = None

from .fsinc.fsinc_1d import sincsq1d

SQRTLOG2 = 0.8325546111576977  # np.sqrt(LOG2)


class Landy19(object):
    """Build a delay Doppler map model Landy19.

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
        grid_space: spatial resolution of the x,y grid (in meters).
        convolution_method: method to perform the convolution. Options are: "sinc2transform" (fast method using
            finufft package, only applicable with ptr_time = sinc^2) or "original" (original Landy et al. 2019 code,
            significantly slower but more flexible).
        use_local_slope: the local slope of the DEM is taken into account to compute the backscatter of the terrain.
            This assumes that the local slope is much larger than the wavelength (generally true) and that this large-scale
            roughness is not taken into account in the small-scale roughness. Set to False to deactivate the local slope.
    """

    def __init__(
        self,
        sensor,
        oversampling_time: int = 4,
        oversampling_doppler: int = 4,
        ptr_time: Optional[str] = None,
        ptr_doppler: Optional[str] = None,
        slant_range_correction: bool | str = True,
        delay_window_widening: int = 1,
        grid_space: float = 10,
        convolution_method: str = "sinc2transform",
        use_local_slope: bool = True,
    ):
        super().__init__()

        self.sensor = sensor
        self.oversampling_time = oversampling_time
        self.oversampling_doppler = oversampling_doppler
        self.use_local_slope = use_local_slope
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

        # L19 Eq 4. There is an error in the text, it's written xi_k but it is xi_0
        self.xi_0 = sensor.wavelength * sensor.pulse_repetition_frequency / (2 * sensor.ndoppler * sensor.velocity)

        # I suspect an error in eq 8. It must be divided by h instead of multiplied.
        # self.K = sensor.wavelength**2 / (4 * np.pi) ** 3 * (C_SPEED * sensor.altitude / 2)  # unit:  m^4 s^-1
        self.K = sensor.wavelength**2 / (4 * np.pi) ** 3 * (C_SPEED / sensor.altitude / 2)  # unit:  m^2 s^-1

        # grid space
        self.grid_space = grid_space  # 10 m should be the default
        max_drang = (
            1.1 * (sensor.ngate - sensor.nominal_gate) / (sensor.pulse_bandwidth * (2 / C_SPEED * sensor.alpha))
        )  # max range (-altitude)
        max_x = np.sqrt(
            (max_drang + sensor.altitude) ** 2 - sensor.altitude**2
        )  # max horizontal extent of the footprint
        self.half_nx = self.half_ny = int(max_x / self.grid_space)

        # by default the PTRs are ft of sinc**2

        self.ptr_doppler = ptr_function(ptr_doppler) if ptr_doppler is not None else sinc2

        if convolution_method == "sinc2transform":
            if finufft is None:
                raise SMRTError(
                    "The package finufft is needed to use the fast sinc2transform convolution."
                    + ' Otherwise use the "original" code of Landy et al. 2018 which is x50-100 slower.'
                )
            if (ptr_time is not None) and (ptr_time != "sinc^2"):
                raise SMRTError(
                    "The fast sinc2transform convolution is only applicable with ptr_time = sinc^2."
                    ' Use the "original" code which is however x50-100 slower'
                )
        elif convolution_method == "original":
            self.ptr_time = ptr_function(ptr_time) if ptr_time is not None else sinc2
            self.tau = delay_sampling_vector(sensor, oversampling_time, self.delay_window_widening)
        else:
            raise SMRTError("Unknown convolution_method. Use sinc2transform or original")
        self.convolution_method = convolution_method

    def delay_doppler_map(self, terrain_info: TerrainInfo):
        return self.delay_doppler_map_with_sigma0func(terrain_info=terrain_info, sigma0=1)

    def delay_doppler_map_with_sigma0func(self, terrain_info: TerrainInfo, sigma0: npt.ArrayLike = 1):
        """Compute the delay doppler map for a surface"""
        sensor = self.sensor

        y, x = np.ogrid[-self.half_ny : self.half_ny, -self.half_nx : self.half_nx]
        x *= self.grid_space
        y *= self.grid_space

        dem = generate_dem(terrain_info, x, y)
        if self.use_local_slope:
            dem_normal = compute_dem_normal(dem, self.grid_space)

        # not used:
        # Lx_doppler = altitude * xi_0  # L19 Eq 15
        # Lx_pulselimited = 2 * np.sqrt(C_SPEED * altitude / (pulse_bandwith * alpha))

        # transform sigma0 in a list for the vectorization
        try:
            sigma0_list = list(sigma0)  # type: ignore
            squeeze_first_dim = False
        except TypeError:
            sigma0_list = [sigma0]
            squeeze_first_dim = True

        ddm = np.zeros(
            (len(sigma0_list), sensor.ngate * self.oversampling_time, sensor.ndoppler * self.oversampling_doppler)
        )

        L_2 = sensor.ndoppler * self.oversampling_doppler // 2
        for ik, k in enumerate(np.arange(-L_2 + 0.5, L_2 + 0.5) / self.oversampling_doppler):
            # the antenna boresight is exactly at the center of the surface, so x0, y0 are the actual antenna position
            # even with mis-pointing

            xi_k = k * self.xi_0

            x0 = sensor.altitude * (xi_k + np.tan(sensor.pitch_angle))
            y0 = sensor.altitude * np.tan(sensor.roll_angle)

            rang_vector = [x - x0, y - y0, dem - sensor.altitude]  # this neglects the Earth curvature

            rho2 = rang_vector[0] ** 2 + rang_vector[1] ** 2  # horizontal distance squared
            rang = np.sqrt(rang_vector[2] ** 2 + rho2 * sensor.alpha)  # L19 eq 10  # this includes the Earth curvature

            # compute local incidence angle
            if self.use_local_slope:
                mu_local_i = -(
                    rang_vector[0] * dem_normal[0] + rang_vector[1] * dem_normal[1] + rang_vector[2] * dem_normal[2]
                ) / np.sqrt(rang_vector[0] ** 2 + rang_vector[1] ** 2 + rang_vector[2] ** 2)

            else:
                #  neglected the local slope
                mu_local_i = -rang_vector[2] / rang

            mu_local_i = np.minimum(mu_local_i, 1.0)

            # L19 Eq 11
            theta_l = np.arctan(-(x - sensor.altitude * xi_k) / rang_vector[2])  # look angle

            # L19 Eq 10  (not used)
            # phi = np.arctan2(y - y0, x - x0)

            # L19 Eq 16 / Note an error in the paper: np.cos(theta) should be np.cos(phi)

            # antenna gain
            # find the angle in the antenna framework
            xg = x - sensor.altitude * xi_k  # simplification of (x0 - sensor.altitude * np.tan(sensor.pitch_angle))
            yg = y  # - (y0 - altitude * np.tan(roll_angle))   # can be simplified because y0 - altiude * tan(rool_angle) == 0 by definition

            xg2 = xg**2
            yg2 = yg**2
            theta_g = np.arctan2(np.sqrt(xg2 + yg2), rang - dem)  # defined in FBEM code

            # the following is keep for reference of the original calculation but it is simplified using
            # cos(arctan(y, x))**2 = x**2 / (x**2 + y**2)
            # phi_g = np.arctan2(y_g, x_g)  # defined in FBEM code
            # cosphi2 = np.cos(phi_g) ** 2
            # sinphi2 = 1 - cosphi2
            # G2 = antenna_gain**2 * np.exp(-2 * theta_g**2 * (cosphi2 / gamma_1**2 + sinphi2 / gamma_2**2))

            cosphi2 = np.zeros_like(theta_g)
            cosphi2 = np.divide(xg2, yg2 + xg2, out=cosphi2, where=(xg2 != 0))

            G2 = sensor.antenna_gain**2 * np.exp(
                -2 * theta_g**2 * (cosphi2 * (1 / self.gamma_1**2 - 1 / self.gamma_2**2) + 1 / self.gamma_2**2)
            )  # here, we used sin² = 1 - cos²

            d = ptr_doppler_landy19(
                theta_l + k * self.xi_0,
                sensor.wavelength,
                sensor.velocity,
                sensor.pulse_repetition_frequency,
                sensor.ndoppler,
            )

            # slant range correction L19 Eq 7  (error 1/altitude -> altitude)
            if self.slant_range_correction == "analytical":
                tau_c = 2 / C_SPEED * np.sqrt((xi_k * sensor.altitude) ** 2 * sensor.alpha + sensor.altitude**2)
            else:
                tau_c = 2 / C_SPEED * sensor.altitude  # not slant range correction

            tau0 = tau_c - 2 / C_SPEED * rang  # L19 Eq 13 (without t, which will be added later)

            tau0 = np.ravel(tau0)

            rang4 = rang**4

            for j, sigma0 in enumerate(sigma0_list):
                if callable(sigma0):
                    # Neglect the diffraction !
                    sigma0_values = sigma0(mu_local_i)
                else:
                    sigma0_values = sigma0

                # L19 Eq 8 integrand without ptr_time
                space_term = self.K * G2 * d * self.grid_space**2 * sigma0_values / rang4  # unit: s^-1
                space_term = np.ravel(space_term)

                if self.convolution_method == "original":
                    # original Landy 2018 method
                    for it, t in enumerate(self.tau):  # after Eq 13
                        tau = t + tau0

                        ptr_t = self.ptr_time(tau, 1 / sensor.pulse_bandwidth)
                        ddm[j, it, ik] += np.dot(ptr_t, space_term)
                elif self.convolution_method == "sinc2transform":
                    # approach optimized by Picard
                    # Calculate the fast sinc^2-transform by ways of the non-uniform fast Fourier transform.
                    # sp = sum sinc^2(x - xp) * s

                    t = np.arange(
                        -sensor.nominal_gate * self.oversampling_time,
                        (sensor.ngate - sensor.nominal_gate) * self.oversampling_time,
                    ) / (sensor.pulse_bandwidth * self.oversampling_time)
                    ddm[j, :, ik] = sincsq1d(
                        tau0 * sensor.pulse_bandwidth, space_term, -t * sensor.pulse_bandwidth, norm=True
                    )  # unit: no
                else:
                    raise RuntimeError("convolution_method not implemented")

        if self.slant_range_correction == "numerical":
            for j in range(len(sigma0_list)):
                ddm[j, :, :] = delay_compensation(
                    sensor, ddm[j, :, :], self.oversampling_time, self.oversampling_doppler
                )

            # print(f'time loop: {time.time()-t0} time convolution: {time.time()-t1}')

        if squeeze_first_dim:
            ddm = ddm[0]
        return ddm


@numba.vectorize(nopython=True, cache=True)
def ptr_doppler_landy19(phi: float, wavelength, velocity, pulse_repetition_frequency, ndoppler):
    k0 = 2 * np.pi / wavelength
    angle = (k0 * velocity / pulse_repetition_frequency) * np.sin(phi)
    return (np.sin(ndoppler * angle) / np.sin(angle)) ** 2 if np.abs(angle) > 1e-6 else float(ndoppler**2)  # unit: no


def compute_dem_normal(dem: np.ndarray[np.floating], grid_space: float):
    """Compute the normal of the dem.

    Args:
        dem: DEM of the terrain
        mu_i: the cosine of the global incidence angle
    Returns:
        vector with the normals
    """
    dzdy, dzdx = np.gradient(dem, grid_space, grid_space)
    # Normal to the slope
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(dzdy)
    norm_n = np.sqrt(nx**2 + ny**2 + nz**2)

    return np.array([nx, ny, nz]) / norm_n


# @numba.vectorize(nopython=True, cache=True)
# def local_incidence_angle(dem_normal, incidence_vector):
#     """Compute the local incidence angle using the dem slop and the local cosine zenith angle

#     Args:
#         dem_normal: normal vector of the DEMgradients: local slop (dz/dy, dz/dz) as output by np.gradients
#         mu_i: the cosine of the global incidence angle
#         phi_i: azimuth of the sensor
#     Returns:
#         the cosine of the local incidence angle
#     """

#     # Produit scalaire et angle local
#     dot_product = nx * ix + ny * iy + nz * iz
#     cos_theta_local = dot_product / (norm_n * norm_i)
#     cos_theta_local = np.clip(cos_theta_local, -1, 1)  # sécurité numérique

#     return theta_local_deg
