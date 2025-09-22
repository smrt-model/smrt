"""This module contains useful functions and classes for rtsolvers"""

from abc import ABCMeta
from collections.abc import Sequence
from typing import Optional, cast

import numpy as np
import scipy.interpolate
import xarray as xr

from smrt.core.error import SMRTError
from smrt.core.result import make_result
from smrt.core.sensor import Sensor
from smrt.core.snowpack import Snowpack
from smrt.rtsolver.streams import compute_stream


class RTSolverBase(metaclass=ABCMeta):
    """RTSolverBase is an abstract class"""

    effective_permittivity: np.ndarray
    substrate_permittivity: np.ndarray
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
        self.substrate_permittivity = (
            self.snowpack.substrate.permittivity(self.sensor.frequency) if self.snowpack.substrate is not None else None
        )

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
    """
    This mixin provides features to deal with discrete ordinates.

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

        self.streams = compute_stream(
            self.n_max_stream, rtsolver.effective_permittivity, rtsolver.substrate_permittivity, mode=self.stream_mode
        )

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

    def add_intensity_mode(self, intensity, intensity_m, m):
        """add intensity mode m to the intensity.

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

        fill_value = np.nan
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

        if np.min(user_mu) < np.min(outmu):
            raise SMRTError(
                "Viewing zenith angle is higher than the stream angles computed by DORT."
                + " Either increase the number of streams or reduce the highest viewing zenith angle."
            )

        # # reverse is necessary for "old" scipy version
        # intfct = scipy.interpolate.interp1d(
        #     outmu[::-1], intensity[::-1, ...], axis=0, fill_value=fill_value, assume_sorted=True
        # )
        # # the previous call could use fill_value to be smart about extrapolation, but it's safer to return NaN (default)

        # # it seems there is a bug in scipy at least when checking the boundary, mu must be sorted
        # # original code that should work: intensity = intfct(mu)
        # i = np.argsort(user_mu)
        # intensity = intfct(user_mu[i])[np.argsort(i)]  # mu[i] sort mu, and [np.argsort(i)] put in back

        intfct = scipy.interpolate.interp1d(outmu, intensity, axis=mu_axis, fill_value=fill_value, assume_sorted=False)
        intensity = intfct(user_mu)

        return intensity

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
    """
    This mixin provides features to deal with coherent layer for RT Solvers.

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
    """
    Return a dict with the profiles of ka, ks, ke and effective permittivity. Can be directly used by Solver to insert
    data in other_data.

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
