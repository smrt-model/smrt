# coding: utf-8

"""The sensor configuration includes all the information describing the sensor viewing geometry (incidence, ...)
and operating parameters (frequency, polarization, ...). The easiest and recommended way to create a :py:class:`~smrt.core.Sensor` instance is
to use one of the convenience functions listed below. The generic functions :py:func:`passive` and :py:func:`active` should cover all the usages,
but functions for specific sensors are more convenient. See examples in the functions documentation below. We recommend to add new sensors/functions here and share your file to be included in SMRT.

.. autofunction:: passive
.. autofunction:: active
"""

from collections.abc import Sequence
import numpy as np

from smrt.core.sensor import Sensor
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

    amsre_frequency_dict = {
        '06': 6.925e9,
        '10': 10.65e9,
        '19': 18.7e9,
        '23': 23.8e9,
        '37': 36.5e9,
        '89': 89e9}

    return common_conical_pmw("AMSR-E", amsre_frequency_dict, channel=channel, frequency=frequency, theta=theta, name='amsre')


def amsr2(channel=None, frequency=None, polarization=None, theta=55):
    """ Configuration for AMSR-2 sensor.

    This function can be used to simulate all 14 AMSR2 channels i.e. frequencies of 6.925, 10.65, 18.7, 23.8, 36.5 and 89 GHz
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

    amsr2_frequency_dict = {
        '06': 6.925e9,
        '07': 7.3e9,
        '10': 10.65e9,
        '19': 18.7e9,
        '23': 23.8e9,
        '37': 36.5e9,
        '89': 89e9}

    return common_conical_pmw("AMSR2", amsr2_frequency_dict, channel=channel, frequency=frequency, theta=theta, name='asmr2')


def cimr(channel=None, frequency=None, polarization=None, theta=55):
    """ Configuration for AMSR-2 sensor.

    This function can be used to simulate all 10 CIMR channels i.e. frequencies of 1.4, 6.9, 10.6, 18.7, 36.5 GHz
    at both polarizations H and V. Alternatively single channels can be specified with 3-character identifiers. 18 and 19 GHz can
    be used interchangably to represent 18.7 GHz, similarly either 36 and 37 can be used to represent the 36.5 GHz channel.
    Note that if you need both H and V polarization (at 37 GHz for instance), use channel="37" instead of channel=["37V", "37H"] 
    as this will result in a more efficient simulation, because most rtsolvers anyway compute both polarizations in one shot.

    :param channel: single channel identifier
    :type channel: 3-character string

    :returns: :py:class:`Sensor` instance
"""

    cimr_frequency_dict = {
        '01': 1.4135e9,
        '06': 6.925e9,
        '10': 10.65e9,
        '19': 18.7e9,
        '37': 36.5e9
    }
    return common_conical_pmw("CIMR", cimr_frequency_dict, channel=channel, frequency=frequency, theta=theta, name='cimr')


def common_conical_pmw(sensor_name, frequency_dict, channel=None, frequency=None, polarization=None, theta=55, name=None):

    if frequency is None:
        # take default values
        frequency = sorted(set(frequency_dict.values()))
    else:
        # recreate the frequency dict
        frequency_dict = {"%02i" % (freq * 1e9): freq for freq in np.atleast_1d(frequency)}

    if polarization is None:
        polarization = ['H', 'V']

    # create the channel map
    channel_map = {freq + pola: dict(frequency=frequency_dict[freq], polarization=pola, theta=theta)
                   for freq in frequency_dict for pola in polarization}

    if channel is not None:
        if isinstance(channel, str):
            channel = [channel]

        # add H and V to channel's name if not present
        new_channel = []
        for ch in channel:
            if ch[-1] not in 'HV':
                new_channel += [ch + 'H', ch + 'V']
            else:
                new_channel += [ch]

        # take into account 18=19 and 36=37
        for ch in new_channel:
            if '18' in ch:
                channel_map[ch] = channel_map.pop('19' + ch[-1])
            if '36' in ch:
                channel_map[ch] = channel_map.pop('37' + ch[-1])

        try:
            channel_map = filter_channel_map(channel_map, new_channel)
        except KeyError:
            raise SMRTError("%s channel not recognized. Expected one of: %s" % (sensor_name, ", ".join(frequency_dict.keys())))

    sensor = passive(channel_map=channel_map, **extract_configuration(channel_map), name=name)

    return sensor


def quikscat(channel=None, theta=None):
    """ Configuration for quikscat sensor.

     This function can be used to simulate the 4 QUIKSCAT channels i.e. incidence angles 46° and 54° and HH and VV polarizations.
     Alternatively a subset of these channels can be specified with 4-character identifiers with polarization first .e.g. HH46, VV54

     :param channel: single channel identifier
     :type channel: 4-character string

     :returns: :py:class:`Sensor` instance
"""

    channel_map = {'HH46': dict(polarization='H', polarization_inc='H', theta=46, theta_inc=46),
                   'VV54': dict(polarization='V', polarization_inc='V', theta=54, theta_inc=54)
                   }

    if channel is None:
        if theta is None:
            theta = [46, 54]

        theta = np.atleast_1d(theta)
        channel = []

        if 46 in theta:
            channel.append('HH46')
        if 54 in theta:
            channel.append('VV54')

    channel_map = filter_channel_map(channel_map, channel)

    if theta is None:
        theta = list({channel_map[ch]['theta'] for ch in channel_map})

    sensor = active(13.4e9, theta,
                    polarization_inc=['V', 'H'], polarization=['V', 'H'],
                    channel_map=channel_map, name='quikscat')

    return sensor


def ascat(theta=None):
    """ Configuration for ASCAT on MetOp satellites.

        Characteristics of the observation configuration: https://ieeexplore.ieee.org/document/7815274

       This function returns a sensor at 5.255 GHz (C-band) and VV polarization. The incidence angle can be chosen or is by defaut from 25° to 65° every 5°

       :param theta: incidence angle (between 25 and 65° in principle)
       :type theta: float or sequence

       :returns: :py:class:`Sensor` instance
  """
    if theta is None:
        theta = np.arange(25, 70, 5)

    channel_map = {('VV%i' % t): dict(polarization_inc='V', polarization='V', theta=t, theta_inc=t) for t in np.atleast_1d(theta)}

    return active(5.255e9, theta,
                  polarization_inc='V', polarization='V',
                  channel_map=channel_map, name='ascat')


def sentinel1(theta=None):
    """ Configuration for C-SAR on Sentinel 1.

       This function return a sensor at 5.405 GHz (C-band). The incidence angle can be chosen or is by defaut from 20 to 45° by step of 5°

       :param theta: incidence angle
       :type theta: float or sequence

       :returns: :py:class:`Sensor` instance
  """
    if theta is None:
        theta = np.arange(20, 46, 5)

    return active(5.405e9, theta,
                  channel_map={channel: dict(polarization=channel[1], polarization_inc=channel[0]) for channel in ['HH', 'VV', 'HV', 'VH']},
                  name='sentinel1')


def smos(theta=None):
    """ Configuration for MIRAS on SMOS.

       This function returns a passive sensor at 1.41 GHz (L-band). The incidence angle can be chosen or is by defaut from 0 to 60° by step of 5°

       :param theta: incidence angle
       :type theta: float or sequence

       :returns: :py:class:`Sensor` instance
  """
    if theta is None:
        theta = np.arange(0, 61, 5)

    channel_map = {'01H': dict(polarization='H', theta=55),
                   '01V': dict(polarization='V', theta=55)
                   }

    return passive(1.41e9, theta, name='smos', channel_map=channel_map)


def smap(mode, theta=40):
    """Configuration for the passive (mode=P) and active (mode=A) sensor on smap

        This function returns either a passive sensor at 1.4 GHz (L-band) sensor or an active sensor at 1.26 GHz. The incidence angle is 40°.

    """

    if mode == 'P':
        return passive(1.4e9, theta=theta, channel_map={pola: dict(polarization=pola) for pola in 'HV'}, name='smap')
    elif mode == 'A':
        return active(1.26e9, theta=theta, theta_inc=theta,
                      channel_map={channel: dict(polarization=channel[1], polarization_inc=channel[0]) for channel in ['HH', 'VV', 'HV']},
                      name='smap')
    else:
        raise SMRTError('mode must by A (active) or P (passive')


def filter_channel_map(channel_map, channel):

    if isinstance(channel, str):
        channel = [channel]
    channel_map = {ch: channel_map[ch] for ch in channel}

    return channel_map


def extract_configuration(channel_map):

    keys = ['frequency', 'polarization', 'theta', 'polarization_inc', 'theta_inc']

    configuration = dict()
    for k in keys:
        try:
            configuration[k] = list({channel_map[ch][k] for ch in channel_map})
        except KeyError:
            continue

    return configuration
