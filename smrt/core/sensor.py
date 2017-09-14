# coding: utf-8

"""The sensor configuration includes all the information describing the sensor viewing geometry (incidence, ...) 
and operating parameters (frequency, polarization, ...). The easiest and recommended way to create a :py:class:`Sensor` instance is 
to use one of the convenience functions such as :py:func:`~smrt.inputs.sensor_list.passive`, :py:func:`~smrt.inputs.sensor_list.active`, :py:func:`~smrt.inputs.sensor_list.amsre`, etc.
Adding a function for a new or unlisted sensor can be done in :py:mod:`~smrt.inputs.sensor_list` if the sensor is common and of general interest.
Otherwise, we recommend to add these functions in your own files (outside of smrt directories).

"""

import numpy as np
import six

# local import
from .error import SMRTError


def passive(frequency, theta, polarization=None):
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

    sensor = Sensor(frequency, None, theta, None, None, polarization)

    sensor.basic_checks()

    return sensor


def active(frequency, theta_inc, theta=None, phi=None, polarization_inc=None, polarization=None):
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

    sensor = Sensor(frequency, theta_inc, theta, phi, polarization_inc, polarization)

    sensor.basic_checks()

    return sensor


class Sensor(object):
    """ Configuration for sensor.
        Use of the functions :py:func:`passive`, :py:func:`active`, or the sensor specific functions
        e.g. :py:func:`amsre` are recommended to access this class.

    """

    def __init__(self, frequency, theta_inc, theta, phi, polarization_inc, polarization):
            """ Build a Sensor. Setting theta_inc to None means passive mode

    :param frequency: Microwave frequency in Hz
    :param theta_inc: zenith angle of incident radiation emitted from the active sensor
    :param polarization_inc. List of single character (H or V) for the incident wave
    :param theta: zenith angle at which the observation is made
    :param theta: azimuth angle at which the observation is made
    :param polarization. List of single character (H or V)

"""
            self.frequency = frequency

            if isinstance(polarization, six.string_types):
                polarization = list(polarization)
            self.polarization = polarization

            if isinstance(polarization_inc, six.string_types):
                polarization_inc = list(polarization_inc)
            self.polarization_inc = polarization_inc

            self.theta = np.atleast_1d(np.radians(theta)).flatten()
            self.mu_s = np.cos(self.theta)

            if phi is not None:
                self.phi = np.atleast_1d(np.radians(phi)).flatten()
            else:
                self.phi = 0.0

            if theta_inc is None:
                self.theta_inc = None
            else:
                self.theta_inc = np.atleast_1d(np.radians(theta_inc)).flatten()
                self.mu_i = np.cos(self.theta_inc)

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

        try:
            # Tests whether frequency is an array
            frequency_min = min(self.frequency)
        except TypeError:
            # Then a single frequency has been specified
            frequency_min = self.frequency
        if frequency_min < 300e6:
            # Checks frequency is above 300 MHz
            raise SMRTError('Frequency not in microwave range: check units are Hz')
