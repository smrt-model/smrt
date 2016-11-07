# coding: utf-8

"""The sensor configuration includes all the information describing the sensor viewing geometry (incidence, ...)
and operating parameters (frequency, polarization, ...). The easiest and recommended way to create a :py:class:`~smrt.core.Sensor` instance is
to use one of the convenience functions listed below. The generic functions :py:func:`passive` and :py:func:`active` should cover all the usages,
but functions for specific sensors are more convenient. See examples in the functions documentation below. We recommend to add new sensors/functions here and share your file to be included in SMRT.

.. autofunction:: passive
.. autofunction:: active
"""

from smrt.core.sensor import Sensor
from smrt.core.error import SMRTError

from smrt.core.sensor import passive, active  # import so they are available from this module


def amsre(channel=None, frequency=None, polarization=None, theta=55):
    """ Configuration for AMSR-E sensor.

    This function can be used to simulate all 12 AMSR-E channels i.e. frequencies of 6.925, 10.65, 18.7, 23.8, 36.5 and 89 GHz
    at both polarizations H and V. Alternatively single channels can be specified with 3-character identifiers. 18 and 19 GHz can
    be used interchangably to represent 18.7 GHz, similarly either 36 and 37 can be used to represent the 36.5 GHz channel.

    :param channel: single channel identifier
    :type channel: 3-character string

    :returns: :py:class:`Sensor` instance

    **Usage example:**

    ::

        from smrt import sensor
        radiometer = sensor.amsre()  # Simulates all channels
        radiometer = sensor.amsre('36V')  # Simulates 36.5 GHz channel only
        radiometer = sensor.amsre('06H')  # 6.925 GHz channel

    """

    amsre_frequency_dict = {
        '06': 6.925e9,
        '10': 10.65e9,
        '18': 18.7e9,
        '23': 23.8e9,
        '36': 36.5e9,
        '89': 89e9}

    if channel is not None:

        if len(channel) == 3:
            polarization = channel[2]
        else:
            polarization = ['H', 'V']

        fch = channel[0:2]

        if fch == "19":       # optional
            fch = "18"        # optional
        if fch == "37":       # optional
            fch = "36"        # optional

        try:
            frequency = amsre_frequency_dict[fch]
        except KeyError:
            raise SMRTError("AMSR-E channel frequency not recognized. Expected one of: 06, 10, 18 or 19, 23, 36 or 37, 89")

    if frequency is None:
        frequency = sorted(amsre_frequency_dict.values())
        polarization = ['H', 'V']

    sensor = Sensor(frequency, None, theta, None, None, polarization)

    return sensor
