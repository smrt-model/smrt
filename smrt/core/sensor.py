# coding: utf-8

"""The sensor configuration includes all the information describing the sensor viewing geometry (incidence, ...) 
and operating parameters (frequency, polarization, ...). The easiest and recommended way to create a :py:class:`Sensor` instance is 
to use one of the convenience functions such as :py:func:`~smrt.inputs.sensor_list.passive`, :py:func:`~smrt.inputs.sensor_list.active`, :py:func:`~smrt.inputs.sensor_list.amsre`, etc.
Adding a function for a new or unlisted sensor can be done in :py:mod:`~smrt.inputs.sensor_list` if the sensor is common and of general interest.
Otherwise, we recommend to add these functions in your own files (outside of smrt directories).

"""

import copy
from collections.abc import Sequence
import numpy as np
from ..core.globalconstants import C_SPEED


# local import
from .error import SMRTError, smrt_warn


def passive(frequency, theta, polarization=None, channel_map=None, name=None):
    """ Generic configuration for passive microwave sensor.

    Return a :py:class:`Sensor` for a microwave radiometer with given frequency, incidence angle and polarization

    :param frequency: frequency in Hz
    :param theta: viewing angle or list of viewing angles in degrees from vertical. Note that some RT solvers compute all
        viewing angles whatever this configuration because it is internally needed part of the multiple scattering calculation.
        It it therefore often more efficient to call the model once with many viewing angles instead of calling it many times
        with a single angle.
    :param polarization: H and/or V polarizations. Both polarizations is the default. Note that most RT solvers compute all
        the polarizations whatever this configuration because the polarizations are coupled in the RT equation.
    :type polarization: list of characters
    :param channel_map: map channel names (keys) to configuration (values). A configuration is a dict with frequency, polarization and other
        such parameters to be used by Result to select the results.
    :type channel_map: dict 
    :param name: name of the sensor
    :type name: string

    :returns: :py:class:`Sensor` instance

    **Usage example:**

    ::

        from smrt import sensor_list
        radiometer = sensor_list.passive(18e9, 50)
        radiometer = sensor_list.passive(18e9, 50, "V")
        radiometer = sensor_list.passive([18e9,36.5e9], [50,55], ["V","H"])

    """

    if polarization is None:
        polarization = ['V', 'H']

    sensor = Sensor(frequency, None, theta, None, None, polarization, channel_map=channel_map, name=name)

    sensor.basic_checks()

    return sensor


def channel_map_for_radar(frequency=None, polarization='HV', order='fp'):
    """
    return a channel_map to convert channel name to frequency and polarization. This function assumes the frequency is coded as a two-digit number
    in GHz with leading 0 if necessary. The polarization is after the frequency if order is 'fp' and before if order is 'pf'.
"""

    if frequency is None:
        frequency_str = ''
    else:
        frequency_str = ['%02i' % np.round(f / 1e9) for f in frequency]

    if order == 'fp':
        def channel_name(freq_str, pola_inc, pola_ref):
            return str(freq_str) + str(pola_inc) + str(pola_ref)
    elif order == 'pf':
        def channel_name(freq_str, pola_inc, pola_ref):
            return str(pola_inc) + str(pola_ref) + str(freq_str)
    else:
        raise SMRTError('order must be fp or pf')

    channel_map = {channel_name(freq_str, pola_inc, pola_refl): dict(frequency=freq, polarization_inc=pola_inc, polarization=pola_refl)
                   for freq_str, freq in zip(frequency_str, frequency) for pola_inc in polarization for pola_refl in polarization}

    return channel_map


def active(frequency, theta_inc, theta=None, phi=None, polarization_inc=None, polarization=None, channel_map=None, name=None):
    """ Configuration for active microwave sensor.

    Return a :py:class:`Sensor` for a radar with given frequency, incidence and viewing angles and polarization

    If polarizations are not specified, quad-pol is the default (VV, VH, HV and HH).
    If the angle of incident radiation is not specified, *backscatter* will be simulated

    :param frequency: frequency in Hz
    :param theta_inc: incident angle in degrees from the vertical
    :param theta: viewing zenith angle in degrees from the vertical. By default, it is equal to theta_inc which corresponds
        to the backscatter direction
    :param phi: viewing azimuth angle in degrees from the incident direction. By default, it is pi which corresponds
        to the backscatter direction
    :param polarization_inc: list of polarizations of the incidence wave ('H' or 'V' or both.)
    :type polarization_inc: list of 1-character strings
    :param polarization: list of viewing polarizations ('H' or 'V' or both)
    :type polarization: list of 1-character strings
    :param channel_map: map channel names (keys) to configuration (values). A configuration is a dict with frequency, polarization and other
        such parameters to be used by Result to select the results.
    :type channel_map: dict
    :param name: name of the sensor
    :type name: string

    :returns: :py:class:`Sensor` instance

    **Usage example:**

    ::

        from smrt import sensor_list
        scatterometer = sensor_list.active(frequency=18e9, theta_inc=50)
        scatterometer = sensor_list.active(18e9, 50, 50, 0, "V", "V")
        scatterometer = sensor_list.active([18e9,36.5e9], theta=50, theta_inc=50, polarization_inc=["V", "H"], polarization=["V", "H"])

    """

    # if polarization is None or polarization == '4P':
    #     polarization = ['VV', 'VH', 'HV', 'HH']

    if theta is None:
        theta = theta_inc

    if phi is None:
        phi = 180.0

    if polarization is None:
        polarization = ['V', 'H']

    if polarization_inc is None:
        polarization_inc = ['V', 'H']

    sensor = Sensor(frequency, theta_inc_deg=theta_inc, theta_deg=theta, phi_deg=phi,
                    polarization_inc=polarization_inc, polarization=polarization,
                    channel_map=channel_map, name=name)

    sensor.basic_checks()

    return sensor


def altimeter(channel, **kwargs):

    return Altimeter(channel=channel, **kwargs)


def make_multi_channel_altimeter(config, channel):
    # helper function to make a single or multi channel altimter sensor object from a config in dict format
    if isinstance(channel, str):
        return altimeter(channel, **config[channel])
    else:
        if channel is None:
            channel = config.keys()
        return SensorList([altimeter(c, **config[c]) for c in channel])


class SensorBase(object):
    pass


class Sensor(SensorBase):
    """ Configuration for sensor.
        Use of the functions :py:func:`passive`, :py:func:`active`, or the sensor specific functions
        e.g. :py:func:`amsre` are recommended to access this class.

    """

    def __init__(self, frequency=None, theta_inc_deg=None, theta_deg=None, phi_deg=None,
                 polarization_inc=None, polarization=None, channel_map=None, name=None, wavelength=None):
        """ Build a Sensor. Setting theta_inc to None means passive mode

    :param frequency: Microwave frequency in Hz
    :param theta_inc_deg: zenith angle in degrees of incident radiation emitted from the active sensor
    :param polarization_inc. List of single character (H or V) for the incident wave
    :param theta_deg: zenith angle in degrees at which the observation is made
    :param phi_deg: azimuth angle at which the observation is made
    :param polarization: List of single character (H or V)
    :param channel_map: map channel names (keys) to configuration (values). A configuration is a dict with frequency, polarization and other
        such parameters to be used by Result to select the results.
    :param name: name of the sensor
    :param wavelength: wavelength of the sensor. Can be set instead of the frequency.
"""
        super().__init__()

        if frequency is not None and wavelength is not None:
            smrt_warn("Sensor requires either frequency or wavelength argument, not both")
        if wavelength is not None:
            self.wavelength = wavelength
        else:
            self.frequency = frequency

        if isinstance(self.frequency, Sequence):
            self.frequency = np.array(self.frequency).squeeze()

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
            self.mu_s = np.cos(self.theta_inc)

    @property
    def wavelength(self):
        if hasattr(self, "_wls"):
            return self._wls  # avoid calculation and numerical rounding error when wavelength has been explicitely set
        else:
            return C_SPEED / self.frequency

    @wavelength.setter
    def wavelength(self, wls):
        self._wls = wls
        self.frequency = C_SPEED / wls

    @property
    def wavenumber(self):
        return 2 * np.pi / self.wavelength

    @property
    def mode(self):
        """returns the mode of observation: "A" for active or "P" for passive.

"""

        if self.theta_inc is None:
            return 'P'
        else:
            return 'A'

    def basic_checks(self):

        # Check frequency range. Below 300 MHz is an indication the units may be wrong
        # Not documented as it will not be called by the user.

        frequency_min = min(np.atleast_1d(self.frequency))

        if frequency_min < 300e6:
            # Checks frequency is above 300 MHz
            smrt_warn('Frequency not in microwave range: check units are Hz')

    def configurations(self):

        for axis in ["frequency", "theta_inc", "polarization_inc", "theta", "phi", "polarization"]:
            values = np.atleast_1d(getattr(self, axis))
            if len(values) > 1:
                yield axis, values

    def iterate(self, axis):
        """Iterate over the configuration for the given axis.

        :param axis: one of the attribute of the sensor (frequency, ...) to iterate along

"""
        values = getattr(self, axis)

        for v in values:
            sensor_subset = copy.copy(self)
            setattr(sensor_subset, axis, v)  # change the sensor values
            yield sensor_subset


class SensorList(SensorBase):

    def __init__(self, sensor_list, axis="channel"):

        super().__init__()

        self.sensor_list = sensor_list
        self.axis = axis

        # check uniqueness of axis
        if axis == 'channel':
            self.channel_list = [ch for s in self.sensor_list for ch in s.channel_map]
            a = self.channel_list
            self.channel_map = {ch: s.channel_map[ch] for s in self.sensor_list for ch in s.channel_map}
        else:
            a = [getattr(s, axis) for s in self.sensor_list]
            self.channel_map = {ch: dict(**s.channel_map[ch], **{axis: getattr(s, axis)}) for s in sensor_list for ch in s.channel_map}

        if None in a:
            raise SMRTError("It is required to set '%s' value for each sensor" % axis)
        if len(set(a)) != len(a):
            raise SMRTError("It is required to set different '%s' values for each sensor" % axis)

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
    """ Configuration for altimeter.
        Use of the functions :py:func:`altimeter`, or the sensor specific functions
        e.g. :py:func:`envisat_ra2` are recommended to access this class.

    """

    def __init__(self, frequency, altitude, beamwidth, pulse_bandwidth, sigma_p=None, off_nadir_angle=0, beam_asymmetry=0,
                 ngate=1024, nominal_gate=40, theta_inc_deg=0., polarization_inc=None, polarization=None, channel=None):

        channel_map = {channel: dict()} if channel is not None else dict()

        super().__init__(frequency=frequency, theta_inc_deg=theta_inc_deg, theta_deg=theta_inc_deg,
                         polarization_inc=polarization_inc, polarization=polarization, channel_map=channel_map)

        self.altitude = altitude
        self.beamwidth = beamwidth
        self.ngate = ngate
        self.pulse_bandwidth = pulse_bandwidth
        self.pulse_sigma = sigma_p if sigma_p is not None else 0.513 / pulse_bandwidth
        self.nominal_gate = nominal_gate
        self.off_nadir_angle = off_nadir_angle
        self.beam_asymmetry = beam_asymmetry
