# coding: utf-8

"""The sensor configuration includes all the information describing the sensor viewing geometry (incidence, ...) 
and operating parameters (frequency, polarization, ...). The easiest and recommended way to create a :py:class:`Sensor` instance is 
to use one of the convenience functions such as :py:func:`~smrt.inputs.sensor_list.passive`, :py:func:`~smrt.inputs.sensor_list.active`, :py:func:`~smrt.inputs.sensor_list.amsre`, etc.
Adding a function for a new or unlisted sensor can be done in :py:mod:`~smrt.inputs.sensor_list` if the sensor is common and of general interest.
Otherwise, we recommend to add these functions in your own files (outside of smrt directories).

"""

import copy
import numpy as np
import six
from ..core.globalconstants import C_SPEED


# local import
from .error import SMRTError


def passive(frequency, theta, polarization=None, channel=None):
    """ Generic configuration for passive microwave sensor.

    Return a :py:class:`Sensor` for a microwave radiometer with given frequency, incidence angle and polarization

    :param frequency: frequency in Hz
    :param theta: viewing angle or list of viewing angles in degrees from vertical. Note that some RT solvers compute all viewing angles whatever this configuration because it is internally needed part of the multiple scattering calculation. It it therefore often more efficient to call the model once with many viewing angles instead of calling it many times with a single angle.
    :param polarization: H and/or V polarizations. Both polarizations is the default. Note that most RT solvers compute all the polarizations whatever this configuration because the polarizations are coupled in the RT equation.
    :type polarization: list of characters

    :returns: :py:class:`Sensor` instance

    **Usage example:**

    ::

        from smrt import sensor
        radiometer = sensor.passive(18e9, 50)
        radiometer = sensor.passive(18e9, 50, "V")
        radiometer = sensor.passive([18e9,36.5e9], [50,55], ["V","H"])

    """

    if polarization is None:
        polarization = ['V', 'H']

    sensor = Sensor(frequency, None, theta, None, None, polarization, channel=channel)

    sensor.basic_checks()

    return sensor


def active(frequency, theta_inc, theta=None, phi=None, polarization_inc=None, polarization=None, channel=None):
    """ Configuration for active microwave sensor.

    Return a :py:class:`Sensor` for a radar with given frequency, incidence and viewing angles and polarization

    If polarizations are not specified, quad-pol is the default (VV, VH, HV and HH).
    If the angle of incident radiation is not specified, *backscatter* will be simulated

    :param frequency: frequency in Hz
    :param theta_inc: incident angle in degrees from the vertical
    :param theta: viewing zenith angle in degrees from the vertical. By default, it is equal to theta_inc which corresponds to the backscatter direction
    :param phi: viewing azimuth angle in degrees from the incident direction. By default, it is pi which corresponds to the backscatter direction
    :param polarization_inc: list of polarizations of the incidence wave ('H' or 'V' or both.)
    :type polarization_inc: list of 1-character strings
    :param polarization: list of viewing polarizations ('H' or 'V' or both)
    :type polarization: list of 1-character strings

    :returns: :py:class:`Sensor` instance

    **Usage example:**

    ::

        from smrt import sensor
        scatterometer = sensor.active(frequency=18e9, theta_inc=50)
        scatterometer = sensor.active(18e9, 50, 50, 0, "V", "V")
        scatterometer = sensor.active([18e9,36.5e9], theta=50, theta_inc=50, polarization_inc=["V", "H"], polarization["V", "H"])

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

    sensor = Sensor(frequency, theta_inc, theta, phi, polarization_inc, polarization, channel=channel)

    sensor.basic_checks()

    return sensor


class SensorBase(object):
    pass


class Sensor(SensorBase):
    """ Configuration for sensor.
        Use of the functions :py:func:`passive`, :py:func:`active`, or the sensor specific functions
        e.g. :py:func:`amsre` are recommended to access this class.

    """

    def __init__(self, frequency=None, theta_inc_deg=None, theta_deg=None, phi_deg=None,
                    polarization_inc=None, polarization=None, channel=None, wavelength=None):
        """ Build a Sensor. Setting theta_inc to None means passive mode

    :param frequency: Microwave frequency in Hz
    :param theta_inc_deg: zenith angle in degrees of incident radiation emitted from the active sensor
    :param polarization_inc. List of single character (H or V) for the incident wave
    :param theta_deg: zenith angle in degrees at which the observation is made
    :param phi_deg: azimuth angle at which the observation is made
    :param polarization: List of single character (H or V)
    :param channel: name of the channel (string)
    :param wavelength

"""
        super().__init__()

        if frequency is not None and wavelength is not None:
            raise SMRTError("Sensor requires either frequency or wavelength argument, not both")
        if wavelength is not None:
            self.wavelength = wavelength
        else:
            self.frequency = frequency

        self.channel = channel

        if isinstance(polarization, six.string_types):
            polarization = list(polarization)
        self.polarization = polarization

        if isinstance(polarization_inc, six.string_types):
            polarization_inc = list(polarization_inc)
        self.polarization_inc = polarization_inc

        if theta_deg is None:
            raise SMRTError("Sensor requires the argument 'theta_deg' to be set")
        self.theta_deg = np.atleast_1d(theta_deg).flatten()

        if len(np.unique(self.theta_deg)) != len(self.theta_deg):
            raise SMRTError("Zenith angle theta has duplicated values which is invalid.")

        self.theta = np.radians(self.theta_deg)
        self.mu_s = np.cos(self.theta)

        if phi_deg is not None:
            self.phi_deg = np.atleast_1d(phi_deg).flatten()
            self.phi = np.radians(self.phi_deg)
        else:
            self.phi = 0.0

        if theta_inc_deg is None:
            self.theta_inc_deg = None
            self.theta_inc = None
        else:
            self.theta_inc_deg = np.atleast_1d(theta_inc_deg).flatten()

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
            raise SMRTError('Frequency not in microwave range: check units are Hz')


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
        self.sensor_list = sensor_list
        self.axis = axis

        # check uniqueness of axis
        l = [getattr(s, axis) for s in self.sensor_list]

        if None in l:
            raise SMRTError("It is required to set '%s' value for each sensor" % axis)
        if len(set(l)) != len(l):
            raise SMRTError("It is required to set different '%s' values for each sensor" % axis)

    @property
    def channel(self):
        return [s.channel for s in self.sensor_list]

    def configurations(self):
        yield self.axis, np.array([getattr(s, self.axis) for s in self.sensor_list])

    def iterate(self, axis=None):

        if axis is not None and axis != self.axis:
            raise SMRTError("SensorList is unable to iterate over a different axis than its axis")
        yield from self.sensor_list

