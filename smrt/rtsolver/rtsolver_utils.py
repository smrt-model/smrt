"""This module contains useful functions and classes for rtsolvers"""

import copy
from abc import ABCMeta
from collections.abc import Sequence
from typing import Optional, cast

import numpy as np
import scipy.interpolate
import xarray as xr

from smrt.core.error import SMRTError
from smrt.core.lib import inverse_planck_function, is_equal_zero, planck_function, smrt_matrix
from smrt.core.result import make_result
from smrt.core.sensor import Sensor
from smrt.core.snowpack import Snowpack
from smrt.rtsolver.streams import compute_stream


class RTSolverBase(metaclass=ABCMeta):
    """RTSolverBase is an abstract class"""

    effective_permittivity: np.ndarray
    # substrate_permittivity: np.ndarray
    sensor: Sensor
    snowpack: Snowpack
    emmodels: list

    def init_solve(self, snowpack, emmodels, sensor, atmosphere=None):
        """Initialize varaibles and perform basic checks.

        Args:
            snowpack: Snowpack object, :py:mod:`smrt.core.snowpack`.
            emmodels: List of electromagnetic models object, :py:mod:`smrt.emmodel`.
            sensor: Sensor object, :py:mod:`smrt.core.sensor`.
            atmosphere: [Optional] Atmosphere object, :py:mod:`smrt.atmosphere`.

        Returns:
            result: Result object, :py:mod:`smrt.core.result.Result`.
        """
        # all these assignements are for convenience, this would break with parallel code (// !!)
        self.emmodels = emmodels
        self.snowpack = snowpack
        self.sensor = sensor

        self.atmosphere = atmosphere

        self.effective_permittivity = np.array([emmodel.effective_permittivity() for emmodel in emmodels])
        # self.substrate_permittivity = (
        #     self.snowpack.substrate.permittivity(self.sensor.frequency) if self.snowpack.substrate is not None else
        # None
        # )

        self.check_sensor()

    def check_sensor(self):
        pass


# class RTSolverBaseLike(Protocol):
#     effective_permittivity: np.ndarray
#     substrate_permittivity: complex
#     streams: Streams
#     sensor: Sensor
#     snowpack: Snowpack
#     emmodels: list


class DiscreteOrdinatesMixin(metaclass=ABCMeta):
    """This mixin provides features to deal with discrete ordinates.

    .. note::

        This mixin defines variables to be used by the declaring class and it assumes some variables exist in the parent
        class (e.g. effective_permittivity, substrate_permittivity). This is valid for a mixin:
        https://stackoverflow.com/questions/36690588/should-mixins-use-parent-attributes
    """

    def init(self, stream_mode="most_refringent", n_max_stream=32, m_max=2):
        self.n_max_stream = n_max_stream
        self.stream_mode = stream_mode
        self.m_max = m_max

    def prepare_streams(self):
        """Compute the streams angle using a quadrature or equivalent"""
        rtsolver = cast(RTSolverBase, self)

        self.streams = compute_stream(self.n_max_stream, rtsolver.effective_permittivity, mode=self.stream_mode)

    def prepare_incident_streams(self) -> list:
        """Compute the streams nearest to the user-requested incident angle (for radar only)."""
        sensor = cast(RTSolverBase, self).sensor
        incident_streams = set()

        for mu_inc in np.cos(sensor.theta_inc):
            i0 = np.searchsorted(-self.streams.outmu, -mu_inc)  # assume mu_inc is sorted in decreasing order.
            if i0 == 0:
                incident_streams.add(i0)
            elif i0 == len(self.streams.outmu):
                incident_streams.add(i0 - 1)
            else:
                incident_streams.add(i0)
                incident_streams.add(i0 - 1)
        incident_streams = sorted(list(incident_streams))  # fix the order (required for the interpolation)

        return incident_streams

    def prepare_incident_intensity(self):
        if self.sensor.mode == "A":
            # send a direct beam

            # incident angle at a given angle
            # use interpolation to get the based effective angle

            #
            # delta(x) = 1/2pi + 1/pi*sum_{n=1}{infinty} cos(nx)
            #

            incident_streams = self.prepare_incident_streams()

            intensity_0 = np.zeros((2 * self.streams.n_air, 2 * len(incident_streams)))
            intensity_higher = np.zeros((3 * self.streams.n_air, 3 * len(incident_streams)))
            # 2 and 3 are for the polarizations

            j0 = 0
            j_higher = 0
            for i in incident_streams:
                power = 1.0 / (2 * np.pi * self.streams.outweight[i])
                for ipol in [0, 1]:
                    intensity_0[2 * i + ipol, j0] = power
                    j0 += 1
                for ipol in [0, 1, 2]:
                    intensity_higher[3 * i + ipol, j_higher] = 2 * power
                    j_higher += 1

        elif self.sensor.mode == "P":
            npol = 2
            incident_streams = []

            if self.atmosphere_result is not None:
                # incident radiation is a function of frequency and incidence angle
                # assume azimuthally symmetric
                intensity_0 = self.atmosphere_result.intensity_down
                assert intensity_0.shape == (npol, self.streams.n_air)
                # convert pola, theta to (theta, pola) and add batch dimension
                intensity_0 = np.swapaxes(intensity_0, 0, 1).flatten()[:, np.newaxis]
                intensity_higher = np.zeros_like(intensity_0)
            else:
                intensity_0 = np.zeros((npol * self.streams.n_air, 1))
                intensity_higher = intensity_0
                intensity_0.flags.writeable = False  # make immutable
                intensity_higher.flags.writeable = False  # make immutable
        else:
            raise SMRTError("Unknow sensor mode")

        return intensity_0, intensity_higher, incident_streams

    def add_intensity_mode(self, intensity, intensity_m, m):
        """Add intensity mode m to the intensity.

        This function assumes that the intensity dimensions are pola, incidence, pola, incidence,...

        """
        sensor = cast(RTSolverBase, self).sensor

        if m == 0:
            if sensor.mode == "P":
                intensity[0:2] += intensity_m[0:2]
            elif sensor.mode == "A":
                intensity[0:2, :, 0:2] += intensity_m[0:2, :, 0:2]
            else:
                raise NotImplementedError()
        else:
            # TODO Ghi: deals with an array of phi
            intensity[0:2] += intensity_m[0:2] * np.cos(m * sensor.phi)
            intensity[2:] += intensity_m[2:] * np.sin(m * sensor.phi)

    def interpolate_intensity(self, outmu, intensity):
        """Interpolate intensity.

        This function assumes that the intensity dimensions are pola, incidence, for the passive mode and pola, pola,
        incidence for the active mode
        """
        sensor = cast(RTSolverBase, self).sensor

        user_mu = np.cos(sensor.theta)  # streams requested by the user

        mu_axis = 1 if sensor.mode == "P" else 2

        if np.max(user_mu) > np.max(outmu):
            imumax = np.argmax(outmu)
            # need extrapolation to 0°
            # add the mean of H and V polarisation for the smallest angle for theta=0 (mu=1)
            if sensor.mode == "P":  # passive
                outmu = np.insert(outmu, 0, 1.0)
                mean_H_V = np.mean(intensity.take(imumax, axis=mu_axis), axis=0)
                intensity = np.insert(intensity, 0, mean_H_V, axis=mu_axis)
            else:  # active
                copol = (intensity[0, 0, imumax] + intensity[1, 1, imumax]) / 2
                crosspol = (intensity[1, 0, imumax] + intensity[0, 1, imumax]) / 2

                intensity = np.insert(
                    intensity,
                    0,
                    [
                        [copol, crosspol, intensity[0, 2, imumax]],
                        [crosspol, copol, intensity[1, 2, imumax]],
                        intensity[2, :, imumax],
                    ],
                    axis=mu_axis,
                )
                outmu = np.insert(outmu, 0, 1.0)

        # safegyard for lowest angle
        # if np.min(user_mu) < np.min(outmu):
        #     raise SMRTError(
        #         "Viewing zenith angle is higher than the stream angles computed by DORT."
        #         + " Either increase the number of streams or reduce the highest viewing zenith angle."
        #     )

        # # reverse is necessary for "old" scipy version
        # intfct = scipy.interpolate.interp1d(
        #     outmu[::-1], intensity[::-1, ...], axis=0, fill_value=fill_value, assume_sorted=True
        # )
        # # the previous call could use fill_value to be smart about extrapolation, but it's safer to return NaN
        # (default)

        # # it seems there is a bug in scipy at least when checking the boundary, mu must be sorted
        # # original code that should work: intensity = intfct(mu)
        # i = np.argsort(user_mu)
        # intensity = intfct(user_mu[i])[np.argsort(i)]  # mu[i] sort mu, and [np.argsort(i)] put in back

        intfct = scipy.interpolate.interp1d(
            outmu, intensity, axis=mu_axis, fill_value="extrapolate", bounds_error=False, assume_sorted=False
        )
        intensity = intfct(user_mu)

        return intensity

    def sum_modes(self, compute_modem, m_max=0):
        if self.sensor.mode == "P":
            npol = 2
        elif self.sensor.mode == "A":
            npol = 3
        else:
            raise NotImplementedError()

        # prepare the atmosphere

        self.atmosphere_result = (
            self.atmosphere.run(
                self.sensor.frequency,
                self.streams.outmu,
                npol,
                rayleigh_jeans_approximation=self.rayleigh_jeans_approximation,
            )
            if self.atmosphere is not None
            else None
        )

        #
        # compute the incident intensity array depending on the sensor

        intensity_0, intensity_higher, incident_streams = (
            self.prepare_incident_intensity()
        )  # TODO Ghi: make an iterator

        #
        # compute the outgoing intensity for each mode
        #
        if self.sensor.mode == "P":
            intensity_up = np.zeros((npol, self.streams.n_air))
        elif self.sensor.mode == "A":
            intensity_up = np.zeros((npol, self.streams.n_air, npol, len(incident_streams)))
            # compute the coherent contribution
            coherent_intensity_up_0 = compute_modem(
                mode=0, streams=self.streams, intensity_down=intensity_0, coherent_only=True
            )
        else:
            raise RuntimeError("unknow sensor mode")

        # sum over the modes

        for m in range(0, m_max + 1):
            intensity_down_m = intensity_0 if m == 0 else intensity_higher

            # compute the upwelling intensity for mode m
            intensity_up_m = compute_modem(
                mode=m,
                streams=self.streams,
                intensity_down=intensity_down_m,
            )

            if self.sensor.mode == "A":
                # substrate the coherent contribution
                intensity_up_m[0:2, :, 0:2, :] -= coherent_intensity_up_0 * (1 + float(m > 0))

            self.add_intensity_mode(intensity_up, intensity_up_m, m)

            # TODO: implement a convergence test if we want to avoid long computation
            # when self.m_max is too high for the phase function.

        if self.sensor.mode == "P":
            if self.atmosphere_result is not None:
                intensity_up = self.atmosphere_result.intensity_up + self.atmosphere_result.transmittance * intensity_up
            intensity_up = self.inverse_planck_function(intensity_up)  # convert back to brightness temperature
            outmu = self.streams.outmu
        elif self.sensor.mode == "A":
            # compress to get only the backscatter
            backscatter_intensity_up = np.empty((npol, npol, len(incident_streams)))
            for j, i in enumerate(incident_streams):
                # the j-th column vector contains the stream i, with angle mu[i]
                # at the same time, convert the dimension to pola, pola, incidence
                backscatter_intensity_up[:, :, j] = intensity_up[:, i, :, j]

            outmu = self.streams.outmu[incident_streams]
            intensity_up = backscatter_intensity_up
        else:
            raise NotImplementedError()

        return outmu, intensity_up

    def make_result(self, outmu, intensity_up, other_coords: Optional[Sequence] = None):
        #  describe the results list of (dimension name, dimension array of value)

        rtsolver = cast(RTSolverBase, self)

        if rtsolver.sensor.mode == "P":
            pola = ["V", "H"]
            coords = [("polarization", pola), ("theta", rtsolver.sensor.theta_deg)]
        else:  # sensor.mode == 'A':
            pola = ["V", "H", "U"]
            coords = [("polarization_inc", pola), ("polarization", pola), ("theta_inc", rtsolver.sensor.theta_inc_deg)]

        if other_coords is not None:
            coords += other_coords

        # store other diagnostic information
        other_data = {
            "stream_angles": xr.DataArray(np.rad2deg(np.arccos(outmu)), coords=[range(len(outmu))]),
        } | prepare_kskaeps_profile_information(
            rtsolver.snowpack, rtsolver.emmodels, effective_permittivity=rtsolver.effective_permittivity, mu=outmu
        )

        return make_result(rtsolver.sensor, intensity_up, coords, other_data=other_data)


class CoherentLayerMixin(metaclass=ABCMeta):
    """This mixin provides features to deal with coherent layer for RT Solvers.

    It provides a function to perform the snowpack transformation.
    """

    def init(self, process_coherent_layers):
        self._process_coherent_layers = process_coherent_layers

    def process_coherent_layers(self):
        if self._process_coherent_layers:
            from smrt.interface.coherent_flat import (
                process_coherent_layers,
            )  # we lazy import this if requested by the users.

            rtsolver = cast(RTSolverBase, self)

            rtsolver.snowpack, rtsolver.emmodels, rtsolver.effective_permittivity = process_coherent_layers(
                rtsolver.snowpack, rtsolver.emmodels, rtsolver.effective_permittivity, rtsolver.sensor
            )
            nlayers = len(rtsolver.emmodels)
            assert len(rtsolver.snowpack.layers) == nlayers
            assert len(rtsolver.snowpack.interfaces) == nlayers
            assert len(rtsolver.effective_permittivity) == nlayers


def prepare_kskaeps_profile_information(snowpack, emmodels, effective_permittivity=None, mu=1):
    """Return a dict with the profiles of ka, ks, ke and effective permittivity. Can be directly used by Solver to
    insert data in other_data.

    ks and ke are the mean in all directions given by mu.

    Args:
        snowpack: the snowpack used for the calculation
        emmodels: the list of instantiated emmodel
        effective_permittivity: list of permittivity. If None, it is obtained from the emmodels
        mu: cosine angles where to compute ks.
    """
    if effective_permittivity is None:
        effective_permittivity = np.array([emmodel.effective_permittivity() for emmodel in emmodels])

    # store other diagnostic information
    layer_index = "layer", range(len(emmodels))
    other_data = {
        "effective_permittivity": xr.DataArray(effective_permittivity, coords=[layer_index]),
        "ks": xr.DataArray([np.mean(em.ks(mu).values) for em in emmodels], coords=[layer_index], name="ks"),
        "ke": xr.DataArray([np.mean(em.ke(mu).values) for em in emmodels], coords=[layer_index], name="ke"),
        "ka": xr.DataArray([getattr(em, "ka", np.nan) for em in emmodels], coords=[layer_index], name="ka"),
        "thickness": xr.DataArray(snowpack.layer_thicknesses, coords=[layer_index], name="thickness"),
    }

    return other_data


class PlanckMixin(metaclass=ABCMeta):
    """This mixin provides features to deal with Planck function.

    .. note::

        This mixin defines variables to be used by the declaring class and it assumes some variables exist in the parent
        class (e.g. sensor). This is valid for a mixin:
        https://stackoverflow.com/questions/36690588/should-mixins-use-parent-attributes
    """

    def init(self, rayleigh_jeans_approximation):
        self.rayleigh_jeans_approximation = rayleigh_jeans_approximation

        if rayleigh_jeans_approximation:
            self.planck_function = lambda tb: tb
            self.inverse_planck_function = lambda radiance: radiance
        else:
            self.planck_function = lambda T: planck_function(self.sensor.frequency, T)
            self.inverse_planck_function = lambda radiance: inverse_planck_function(self.sensor.frequency, radiance)


def compute_interface_properties(
    frequency, interfaces, substrate, permittivity, streams, m_max, npol, compress=True, auto_reduce_npol=True
):
    """Compute the reflection adn transmission for each interface and store them in a InterfaceProperties object."""
    intprop = InterfaceProperties(compress=compress, auto_reduce_npol=auto_reduce_npol)
    intprop.compute(frequency, interfaces, substrate, permittivity, streams, m_max, npol)
    return intprop


class InterfaceProperties(object):
    def __init__(self, layer=None, mode=None, compress=True, auto_reduce_npol=True, coherent_only=False):
        self.selected_layer = layer
        self.selected_mode = mode
        self.compress = compress
        self.auto_reduce_npol = auto_reduce_npol
        self.coherent_only = coherent_only

        self.Rtop_coh = dict()
        self.Rtop_diff = dict()
        self.Ttop_coh = dict()
        self.Ttop_diff = dict()
        self.Rbottom_coh = dict()
        self.Rbottom_diff = dict()
        self.Tbottom_coh = dict()
        self.Tbottom_diff = dict()

    def sel(self, layer=None, mode=None, coherent_only=False):
        intprop = copy.copy(self)  # shallow copy is enough since we don't modify the dicts
        intprop.selected_layer = self.selected_layer if layer is None else layer
        intprop.selected_mode = mode if mode is not None else self.selected_mode
        intprop.coherent_only = coherent_only
        return intprop

    def compute(self, frequency, interfaces, substrate, permittivity, streams, m_max, npol):
        nlayer = len(interfaces)
        for layer in range(nlayer):
            eps_lm1 = permittivity[layer - 1] if layer > 0 else 1
            eps_l = permittivity[layer]
            eps_lp1 = permittivity[layer + 1] if layer < nlayer - 1 else None

            # compute reflection coefficient between layer l and l - 1  UP
            # snow-snow UP
            self.Rtop_coh[layer] = interfaces[layer].specular_reflection_matrix(
                frequency, eps_l, eps_lm1, streams.mu[layer], npol
            )

            self.Rtop_diff[layer] = (
                normalize_diffuse_matrix(
                    interfaces[layer].ft_even_diffuse_reflection_matrix(
                        frequency, eps_l, eps_lm1, streams.mu[layer], streams.mu[layer], m_max, npol
                    ),
                    streams.mu[layer],
                    streams.mu[layer],
                    streams.weight[layer],
                )
                if not self.coherent_only and hasattr(interfaces[layer], "ft_even_diffuse_reflection_matrix")
                else smrt_matrix(0)
            )

            # compute transmission coefficient between l and l - 1 UP
            # snow-snow or air UP
            self.Ttop_coh[layer] = interfaces[layer].coherent_transmission_matrix(
                frequency, eps_l, eps_lm1, streams.mu[layer], npol
            )

            mu_t = streams.mu[layer - 1] if layer > 1 else streams.outmu

            self.Ttop_diff[layer] = (
                normalize_diffuse_matrix(
                    interfaces[layer].ft_even_diffuse_transmission_matrix(
                        frequency, eps_l, eps_lm1, mu_t, streams.mu[layer], m_max, npol
                    )
                    * (eps_l.real / eps_lm1.real),
                    mu_t,
                    streams.mu[layer],
                    streams.weight[layer],
                )
                if not self.coherent_only and hasattr(interfaces[layer], "ft_even_diffuse_transmission_matrix")
                else smrt_matrix(0)
            )

            # compute transmission coefficient between l and l + 1  DOWN
            if layer < nlayer - 1:
                # snow-snow DOWN
                self.Tbottom_coh[layer] = interfaces[layer + 1].coherent_transmission_matrix(
                    frequency, eps_l, eps_lp1, streams.mu[layer], npol
                )
                self.Tbottom_diff[layer] = (
                    normalize_diffuse_matrix(
                        interfaces[layer + 1].ft_even_diffuse_transmission_matrix(
                            frequency, eps_l, eps_lp1, streams.mu[layer + 1], streams.mu[layer], m_max, npol
                        )
                        * (eps_l.real / eps_lp1.real),
                        streams.mu[layer + 1],
                        streams.mu[layer],
                        streams.weight[layer],
                    )
                    if not self.coherent_only and hasattr(interfaces[layer + 1], "ft_even_diffuse_transmission_matrix")
                    else smrt_matrix(0)
                )

            elif substrate is not None:
                # sub-snow
                self.Tbottom_coh[nlayer - 1] = substrate.emissivity_matrix(frequency, eps_l, streams.mu[layer], npol)
                self.Tbottom_diff[nlayer - 1] = smrt_matrix(0)
            else:
                # sub-snow
                self.Tbottom_coh[nlayer - 1] = smrt_matrix(0)
                self.Tbottom_diff[nlayer - 1] = smrt_matrix(0)

            # compute reflection coefficient between l and l + 1  DOWN
            if layer < nlayer - 1:
                # snow-snow DOWN
                self.Rbottom_coh[layer] = interfaces[layer + 1].specular_reflection_matrix(
                    frequency, eps_l, eps_lp1, streams.mu[layer], npol
                )

                self.Rbottom_diff[layer] = (
                    normalize_diffuse_matrix(
                        interfaces[layer + 1].ft_even_diffuse_reflection_matrix(
                            frequency, eps_l, eps_lp1, streams.mu[layer], streams.mu[layer], m_max, npol
                        ),
                        streams.mu[layer],
                        streams.mu[layer],
                        streams.weight[layer],
                    )
                    if not self.coherent_only and hasattr(interfaces[layer + 1], "ft_even_diffuse_reflection_matrix")
                    else smrt_matrix(0)
                )

            elif substrate is not None:
                # snow-substrate
                self.Rbottom_coh[layer] = substrate.specular_reflection_matrix(
                    frequency, eps_l, streams.mu[layer], npol
                )
                if not self.coherent_only:
                    self.Rbottom_diff[layer] = (
                        normalize_diffuse_matrix(
                            substrate.ft_even_diffuse_reflection_matrix(
                                frequency, eps_l, streams.mu[layer], streams.mu[layer], m_max, npol
                            ),
                            streams.mu[layer],
                            streams.mu[layer],
                            streams.weight[layer],
                        )
                        if hasattr(substrate, "ft_even_diffuse_reflection_matrix")
                        else smrt_matrix(0)
                    )

            else:
                self.Rbottom_coh[layer] = smrt_matrix(0)  # fully transparent substrate
                self.Rbottom_diff[layer] = smrt_matrix(0)

        # air-snow DOWN
        self.Tbottom_coh[-1] = interfaces[0].coherent_transmission_matrix(
            frequency, 1, permittivity[0], streams.outmu, npol
        )

        self.Tbottom_diff[-1] = (
            normalize_diffuse_matrix(
                interfaces[0].ft_even_diffuse_transmission_matrix(
                    frequency, 1, permittivity[0], streams.mu[0], streams.outmu, m_max, npol
                )
                / permittivity[0].real,
                streams.mu[0],
                streams.outmu,
                streams.outweight,
            )
            if not self.coherent_only and hasattr(interfaces[0], "ft_even_diffuse_transmission_matrix")
            else smrt_matrix(0)
        )

        # air-snow DOWN
        self.Rbottom_coh[-1] = interfaces[0].specular_reflection_matrix(
            frequency, 1, permittivity[0], streams.outmu, npol
        )
        self.Rbottom_diff[-1] = (
            normalize_diffuse_matrix(
                interfaces[0].ft_even_diffuse_reflection_matrix(
                    frequency, 1, permittivity[0], streams.outmu, streams.outmu, m_max, npol
                ),
                streams.outmu,
                streams.outmu,
                streams.outweight,
            )
            if not self.coherent_only and hasattr(interfaces[0], "ft_even_diffuse_reflection_matrix")
            else smrt_matrix(0)
        )

    def reflection_top(self, layer=None, mode=None, coherent_only=False):
        lay = self.selected_layer if layer is None else layer
        m = self.selected_mode if mode is None else mode
        return self.combine_coherent_diffuse_matrix(self.Rtop_coh[lay], self.Rtop_diff[lay], m, coherent_only)

    def specular_reflection_top(self, layer=None, mode=None):
        lay = self.selected_layer if layer is None else layer
        m = self.selected_mode if mode is None else mode
        return self.combine_coherent_diffuse_matrix(self.Rtop_coh[lay], self.Rtop_diff[lay], m, coherent_only=True)

    def reflection_bottom(self, layer=None, mode=None, coherent_only=False):
        lay = self.selected_layer if layer is None else layer
        m = self.selected_mode if mode is None else mode
        return self.combine_coherent_diffuse_matrix(self.Rbottom_coh[lay], self.Rbottom_diff[lay], m, coherent_only)

    def specular_reflection_bottom(self, layer=None, mode=None):
        lay = self.selected_layer if layer is None else layer
        m = self.selected_mode if mode is None else mode
        return self.combine_coherent_diffuse_matrix(
            self.Rbottom_coh[lay], self.Rbottom_diff[lay], m, coherent_only=True
        )

    def transmission_top(self, layer=None, mode=None, coherent_only=False):
        lay = self.selected_layer if layer is None else layer
        m = self.selected_mode if mode is None else mode
        return self.combine_coherent_diffuse_matrix(self.Ttop_coh[lay], self.Ttop_diff[lay], m, coherent_only)

    def coherent_transmission_top(self, layer=None, mode=None):
        lay = self.selected_layer if layer is None else layer
        m = self.selected_mode if mode is None else mode
        return self.combine_coherent_diffuse_matrix(self.Ttop_coh[lay], self.Ttop_diff[lay], m, coherent_only=True)

    def transmission_bottom(self, layer=None, mode=None, coherent_only=False):
        lay = self.selected_layer if layer is None else layer
        m = self.selected_mode if mode is None else mode
        return self.combine_coherent_diffuse_matrix(self.Tbottom_coh[lay], self.Tbottom_diff[lay], m, coherent_only)

    def coherent_transmission_bottom(self, layer=None, mode=None):
        lay = self.selected_layer if layer is None else layer
        m = self.selected_mode if mode is None else mode
        return self.combine_coherent_diffuse_matrix(
            self.Tbottom_coh[lay], self.Tbottom_diff[lay], m, coherent_only=True
        )

    def combine_coherent_diffuse_matrix(self, mat_coh, mat_diff, m, coherent_only):
        if self.compress:
            mat_coh = mat_coh.compress(mode=m, auto_reduce_npol=self.auto_reduce_npol)

        if (not coherent_only) and (not is_equal_zero(mat_diff)):
            # the coef comes from the integration of \int dphi' cos(m (phi-phi')) cos(n phi')
            # m=n=0 --> 2*np.pi
            # m=n > 1 --> np.pi
            if m == 0:
                coef = 2 * np.pi
                # npol = 2
            else:
                coef = np.pi  # the factor 2*np.pi comes from the integration of \int dphi
                # npol = 3

            if self.compress:
                mat_diff = mat_diff.compress(mode=m, auto_reduce_npol=self.auto_reduce_npol)
            return coef * mat_diff + mat_coh
        else:
            return mat_coh


def normalize_diffuse_matrix(mat, mu_st, mu_i, weights):
    if is_equal_zero(mat):
        return mat

    if mat.mtype == "dense5":
        mat *= mu_i * weights  # the last dimension
        mat /= mu_st[:, np.newaxis]  # before the last dimension
    elif mat.mtype == "diagonal5":
        if mu_i is mu_st:
            mat *= weights
        else:
            mat *= mu_i * weights / mu_st  # the last dimension
    return mat
