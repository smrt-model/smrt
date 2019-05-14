# coding: utf-8

"""The sensor configuration includes all the information describing the sensor viewing geometry (incidence, ...)
and operating parameters (frequency, polarization, ...). The easiest and recommended way to create a :py:class:`~smrt.core.Sensor` instance is
to use one of the convenience functions listed below. The generic functions :py:func:`passive` and :py:func:`active` should cover all the usages,
but functions for specific sensors are more convenient. See examples in the functions documentation below. We recommend to add new sensors/functions here and share your file to be included in SMRT.

.. autofunction:: passive
.. autofunction:: active
"""

import six
from collections import Sequence

from smrt.core.sensor import Sensor, SensorList
from smrt.core.error import SMRTError


from smrt.core.sensor import passive, active  # import so they are available from this module


def amsre(channel=None, frequency=None, polarization=None, theta=55):
    """ Configuration for AMSR-E sensor.

    This function can be used to simulate all 12 AMSR-E channels i.e. frequencies of 6.925, 10.65, 18.7, 23.8, 36.5 and 89 GHz
    at both polarizations H and V. Alternatively single channels can be specified with 3-character identifiers. 18 and 19 GHz can
    be used interchangably to represent 18.7 GHz, similarly either 36 and 37 can be used to represent the 36.5 GHz channel.
    Note that if you need both H and V polarization (at 37 GHz for instance), use channel="37" instead of channel=["37V", "37H"] 
    as this will result in a more efficient simulation, because most rtsolvers anyway compute both polarizations in one shot.

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
    if isinstance(channel, Sequence) and not isinstance(channel, six.string_types):
        if frequency is not None:
            raise SMRTError("Either channel or frequency should be given. Mixing both arguments is not understood.")
        return SensorList([amsre(c, frequency=None, polarization=polarization, theta=theta) for c in channel])


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

    sensor = Sensor(frequency, None, theta, None, None, polarization, channel)

    return sensor


def quickscat(channel=None, theta=None, polarization=None):
    """ Configuration for quickscat sensor.

     This function can be used to simulate the 4 QUICKSCAT channels i.e. incidence angles 46° and 54° and HH and VV polarizations.
     Alternatively a subset of these channels can be specified with 4-character identifiers with polarization first .e.g. HH46, VV54

     :param channel: single channel identifier
     :type channel: 4-character string

     :returns: :py:class:`Sensor` instance
"""
    if channel is None:
        if theta is None:
            theta = [46, 54]

        if polarization is None:
            polarization = polarization_inc = ['V', 'H']
        else:
            polarization_inc = polarization[1]
            polarization = polarization[0]

    else:

        t, theta, polarization, polarization_inc = decompose_channel(channel, (0, 2, 2))

    sensor = active(13.4e9, theta, polarization_inc=polarization_inc, polarization=polarization)

    return sensor

def ascat(theta=None):
    """ Configuration for ASCAT on ENVISAT sensor.

       This function return a sensor at 5.255 GHz (C-band) and VV polarization. The incidence angle can be chosen or is by defaut from 25° to 65° every 5°

       :param theta: incidence angle (between 25 and 65° in principle)
       :type theta: float or sequence

       :returns: :py:class:`Sensor` instance
  """
    if theta is None:
        theta = np.arange(25, 70, 5)

    return active(5.255e9, theta, polarization_inc='V', polarization='V')




def decompose_channel(channel, lengths):

    if isinstance(channel, Sequence) and not isinstance(channel, six.string_types):

        data = [decompose_channel(ch) for ch in channel]
        frequency, theta, polarization, polarization_inc = tuple(map(list, zip(*data)))  # transpose

    else:
        if len(channel) !=  sum(lengths):
            raise SMRTError("the channel has an incorrect length")
        if lengths[0] > 0:
            frequency = channel[0:lengths[0]]
        else:
            frequency = None

        if lengths[1] > 0:
            polarization = channel[lengths[0]]
            if lengths[1] == 2:
                polarization_inc = channel[lengths[0]+1]
        else:
            polarization_inc = polarization = None

        if lengths[2] > 0:
            theta = float(channel[-lengths[2]:])
        else:
            theta = None

    return frequency, theta, polarization, polarization_inc