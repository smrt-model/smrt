# coding: utf-8

"""
This module defines the configuration for sensors used in radiative transfer simulations. The sensor configuration
includes all the information describing the sensor viewing geometry (incidence, ...) and operating parameters
(frequency, polarization, ...). The easiest and recommended way to create a :py:class:`Sensor` instance is to use one of
the convenience functions such as :py:func:`~smrt.inputs.sensor_list.passive`,
:py:func:`~smrt.inputs.sensor_list.active`, :py:func:`~smrt.inputs.sensor_list.amsre`, etc. Adding a function for a new
or unlisted sensor can be done in :py:mod:`~smrt.inputs.sensor_list` if the sensor is common and of general interest.
Otherwise, we recommend to add these functions in your own files (outside of smrt directories).
"""

import copy
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..core.globalconstants import C_SPEED, EARTH_RADIUS

# local import
from .error import SMRTError, smrt_warn


class SensorBase(object):
    pass


ANGULAR_WAVENUMBER = 2 * np.pi / C_SPEED


class Sensor(SensorBase):
    """
    This class contains a basic sensor configuration.

    Use of the functions :py:func:`passive`, :py:func:`active`, or the sensor specific functions
    e.g. :py:func:`amsre` are recommended to access this class.
    """

    frequency: npt.ArrayLike
    theta_inc: npt.ArrayLike
    theta: npt.ArrayLike
    phi: npt.ArrayLike
    polarization_inc: str
    polarization: str
    channel_map: dict
    name: str

    def __init__(
        self,
        frequency=None,
        theta_inc_deg=None,
        theta_deg=None,
        phi_deg=None,
        polarization_inc=None,
        polarization=None,
        channel_map=None,
        name=None,
        wavelength=None,
    ):
        """
        Build a Sensor.

        Setting theta_inc to None means passive mode

        Args:
            frequency: frequency in Hz.
            theta_inc_deg: zenith angle in degrees of incident radiation emitted from the active sensor.
            polarization_inc: List of single character (H or V) for the incident wave.
            theta_deg: zenith angle in degrees at which the observation is made.
            phi_deg: azimuth angle at which the observation is made.
            polarization: List of single character (H or V).
            channel_map: map channel names (keys) to configuration (values). A configuration is a dict with frequency,
                polarization and other such parameters to be used by Result to select the results.
            name: name of the sensor.
            wavelength: wavelength of the sensor. Can be set instead of the frequency.
        """
        super().__init__()

        if frequency is not None:
            if wavelength is not None:
                smrt_warn("Sensor requires either frequency or wavelength argument, not both")

            self.frequency = np.asarray(frequency).squeeze() if isinstance(frequency, Sequence) else frequency
        elif wavelength is not None:
            wavelength = np.asarray(wavelength).squeeze() if isinstance(wavelength, Sequence) else wavelength
            self.frequency = C_SPEED / wavelength
        else:
            raise SMRTError("Either frequency or wavelength is required")

        self.channel_map = channel_map or dict()

        self.name = name

        if isinstance(polarization, str):
            polarization = list(polarization)
        self.polarization = polarization

        if isinstance(polarization_inc, str):
            polarization_inc = list(polarization_inc)
        self.polarization_inc = polarization_inc

        if theta_deg is None:
            raise SMRTError("Sensor requires the argument 'theta_deg' to be set")
        self.theta_deg = np.atleast_1d(theta_deg).flatten().astype(dtype=float)

        if len(np.unique(self.theta_deg)) != len(self.theta_deg):
            raise SMRTError("Zenith angle theta has duplicated values which is invalid.")

        self.theta = np.radians(self.theta_deg)
        self.mu_s = np.cos(self.theta)

        if phi_deg is not None:
            self.phi_deg = np.atleast_1d(phi_deg).flatten().astype(dtype=float)
            self.phi = np.radians(self.phi_deg)
        else:
            self.phi = 0.0

        if theta_inc_deg is None:
            self.theta_inc_deg = None
            self.theta_inc = None
        else:
            self.theta_inc_deg = np.atleast_1d(theta_inc_deg).flatten().astype(dtype=float)

            if len(np.unique(self.theta_inc_deg)) != len(self.theta_inc_deg):
                raise SMRTError("Zenith angle theta_inc has duplicated values which is invalid.")

            self.theta_inc = np.radians(self.theta_inc_deg)
            self.mu_i = np.cos(self.theta_inc)

    @property
    def wavelength(self):
        return C_SPEED / self.frequency

    @property
    def wavenumber(self):
        return ANGULAR_WAVENUMBER * self.frequency

    @property
    def mode(self):
        """
        Return the mode of observation: "A" for active or "P" for passive.
        """

        if self.theta_inc is None:
            return "P"
        else:
            return "A"

    def basic_checks(self):
        # Check frequency range. Below 300 MHz is an indication the units may be wrong
        # Not documented as it will not be called by the user.

        frequency_min = np.min(np.atleast_1d(self.frequency))

        if frequency_min < 100e6:
            # Checks frequency is above 100 MHz
            smrt_warn("Frequency not in microwave range: check units are Hz")

    def configurations(self):
        for axis in [
            "frequency",
            "theta_inc",
            "polarization_inc",
            "theta",
            "phi",
            "polarization",
        ]:
            values = np.atleast_1d(getattr(self, axis))
            if len(values) > 1:
                yield axis, values

    def iterate(self, axis):
        """
        Iterate over the configuration for the given axis.

        Args:
            axis: one of the attribute of the sensor (frequency, ...) to iterate along
        """
        values = getattr(self, axis)

        for v in values:
            sensor_subset = copy.copy(self)
            setattr(sensor_subset, axis, v)  # change the sensor values
            yield sensor_subset


class SensorList(SensorBase):
    sensor_list: list[Sensor]

    def __init__(self, sensor_list, axis="channel"):
        super().__init__()

        self.sensor_list = sensor_list
        self.axis = axis

        # check uniqueness of axis
        if axis == "channel":
            self.channel_list = [ch for s in self.sensor_list for ch in s.channel_map]
            a = self.channel_list
            self.channel_map = {ch: s.channel_map[ch] for s in self.sensor_list for ch in s.channel_map}
        else:
            a = [getattr(s, axis) for s in self.sensor_list]
            self.channel_map = {
                ch: dict(**s.channel_map[ch], **{axis: getattr(s, axis)}) for s in sensor_list for ch in s.channel_map
            }

        if None in a:
            raise SMRTError(f"It is required to set '{axis}' value for each sensor")
        if len(set(a)) != len(a):
            raise SMRTError(f"It is required to set different '{axis}' values for each sensor")

    @property
    def channel(self):
        return [ch for s in self.sensor_list for ch in s.channel_map]

    @property
    def frequency(self):
        return [s.frequency for s in self.sensor_list]

    def configurations(self):
        if self.axis == "channel":
            yield self.axis, np.array(self.channel_list)
        else:
            yield self.axis, np.array([getattr(s, self.axis) for s in self.sensor_list])

    def iterate(self, axis=None):
        if axis is not None and axis != self.axis:
            raise SMRTError("SensorList is unable to iterate over a different axis than its axis")
        yield from self.sensor_list


class Altimeter(Sensor):
    """Configuration for LRM and SAR altimeters.
    Use of the functions :py:func:`sar_altimeter`, or the sensor specific functions
    e.g. :py:func:`sentinel3_sarm` are recommended to access this class.

    """

    def __init__(
        self,
        frequency,
        altitude,
        pulse_bandwidth,
        beamwidth_alongtrack,
        beamwidth_acrosstrack,
        pulse_repetition_frequency=0,  # can be zero for LRM, but not for SAR
        velocity=0,  # can be zero for LRM, but not for SAR
        antenna_gain=1,
        ngate=128,
        ndoppler=0,  # 0 = LRM
        nominal_gate=40,
        doppler_window="rect",
        pitch_angle_deg=0.0,
        roll_angle_deg=0.0,
        theta_inc_deg=0.0,
        polarization_inc=None,
        polarization=None,
        channel=None,
    ):
        """Build a SAR altimeter sensor configuration.


        Args:
            frequency: frequency in Hz.
            altitude: altitude of the sensor in m.
            pulse_bandwidth: pulse bandwidth in Hz.
            beamwidth_alongtrack: beamwidth along track in degrees.
            beamwidth_acrosstrack: beamwidth across track in degrees.
            pulse_repetition_frequency: pulse repetition frequency in Hz (SAR mode). Can be zero for LRM mode.
            velocity: velocity of the sensor in m/s (SAR mode). Can be unset or zero for LRM mode.
            antenna_gain: one-way antenna gain at the center of the antenna (unitless).
            ngate: number of range gates.
            ndoppler: number of Doppler bins (SAR mode). Must be 0 for LRM mode.
            nominal_gate: nominal gate number (used for georeferencing).
            doppler_window: Doppler window type ('rect' or 'hamming').
            pitch_angle_deg: pitch angle in degrees.
            roll_angle_deg: roll angle in degrees.
            theta_inc_deg: incidence angle in degrees from nadir.
            polarization_inc: list of single character (H or V) for the incident wave.
            polarization: list of single character (H or V) for the received wave.
            channel: name of the channel.
        """

        channel_map = {channel: dict()} if channel is not None else dict()

        super().__init__(
            frequency=frequency,
            theta_inc_deg=theta_inc_deg,
            theta_deg=theta_inc_deg,
            polarization_inc=polarization_inc,
            polarization=polarization,
            channel_map=channel_map,
            phi_deg=180,  # this is important to get backscatter with DORT
        )

        self.altitude = altitude
        self.beamwidth_alongtrack = beamwidth_alongtrack
        self.beamwidth_acrosstrack = beamwidth_acrosstrack
        self.antenna_gain = antenna_gain
        self.pulse_repetition_frequency = pulse_repetition_frequency
        self.velocity = velocity
        self.ngate = ngate
        self.ndoppler = ndoppler
        self.pulse_bandwidth = pulse_bandwidth
        self.nominal_gate = nominal_gate
        self.pitch_angle = np.deg2rad(pitch_angle_deg)
        self.roll_angle = np.deg2rad(roll_angle_deg)

        self.doppler_window = doppler_window

        # Earth sphericity compensation # see Chelton 1989
        self.alpha = 1 + altitude / EARTH_RADIUS  # earth sphericity compensation # see Chelton 1989

    @property
    def burst_duration(self):
        # used by some delay_doppler_map models:
        return self.ndoppler / self.pulse_repetition_frequency

    @property
    def off_nadir_angle(self):
        # off nadir angle in radians
        return np.arccos(np.cos(self.pitch_angle) * np.cos(self.roll_angle))
