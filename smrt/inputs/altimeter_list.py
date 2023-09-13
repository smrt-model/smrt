

from smrt.core.sensor import make_multi_channel_altimeter, altimeter
from smrt.core.error import SMRTError

#
# List of sensors
#


def envisat_ra2(channel=None):
    """return an Altimeter instance for the ENVISAT RA2 altimeter.

    :param channel: can be 'S', 'Ku', or both. Default is both.

"""

    config = {
        'Ku': dict(frequency=13.575e9,
                   altitude=800e3,
                   pulse_bandwidth=320e6,
                   ngate=128,
                   nominal_gate=45,
                   beamwidth=1.29,
                   ),
        'S': dict(frequency=3.2e9,
                  altitude=800e3,
                  pulse_bandwidth=160e6,
                  ngate=128,
                  nominal_gate=32,    # to correct, the value is rather close to 25
                  beamwidth=5.5,  # Lacroix et al. and Fatras et al.,
                  ),
    }

    return make_multi_channel_altimeter(config, channel)


def sentinel3_sral(channel=None):
    """return an Altimeter instance for the Sentinel 3 SRAL instrument.

    :param channel: can be 'Ku' only ('C' is to be implemented)

"""

    config = {
        'Ku': dict(frequency=13.575e9,
                   altitude=814e3,
                   pulse_bandwidth=320e6,
                   nominal_gate=44,
                   ngate=128,
                   beamwidth=1.35,
                   ),
    }

    return make_multi_channel_altimeter(config, channel)


def saral_altika():
    """return an Altimeter instance for the Saral/AltiKa instrument.

"""

    params = dict(frequency=35.75e9,
                  altitude=800e3,
                  pulse_bandwidth=480e6,
                  nominal_gate=51,
                  ngate=128,
                  beamwidth=0.605,
                  )
    return altimeter(channel='Ka', **params)


def cryosat2_lrm():
    """ Return an altimeter instance for CryoSat-2

    Parameters from https://earth.esa.int/web/eoportal/satellite-missions/c-missions/cryosat-2
    Altitude from https://doi.org/10.1016/j.asr.2018.04.014
    Beam width is 1.08 along track and 1.2 across track

    """

    params = dict(frequency=13.575e9,
                  altitude=720e3,
                  pulse_bandwidth=320e6,
                  nominal_gate=50,  # Estimate - needs better definition
                  ngate=128,
                  beamwidth=1.2,
                  )
    return altimeter(channel='Ku', **params)


def cryosat2_sin():
    """ Return an altimeter instance for CryoSat-2: SIN mode

    Parameters from https://earth.esa.int/web/eoportal/satellite-missions/c-missions/cryosat-2
    Altitude from https://doi.org/10.1016/j.asr.2018.04.014
    Beam width is 1.08 along track and 1.2 across track. Unfortunately SMRT is not able to take into
    account elliptical footprints yet.

    """

    params = dict(frequency=13.575e9,
                  altitude=720e3,
                  pulse_bandwidth=320e6,
                  nominal_gate=164,  # Estimate - needs better definition
                  ngate=512,
                  beamwidth=1.2,
                  )
    return altimeter(channel='Ku', **params)


def asiras_lam(altitude=None):
    """ Return an altimeter instance for ASIRAS in Low Altitude Mode

    Parameters from https://earth.esa.int/web/eoportal/airborne-sensors/asiras
    Beam width is 2.2 x 9.8 deg
    """
    if altitude is None:
        raise SMRTError('Aircraft altitude must be defined')
    else:
        altitude = altitude

    params = dict(frequency=13.5e9,
                  pulse_bandwidth=1e9,
                  altitude=altitude,
                  nominal_gate=41,  # Estimate - needs better definition
                  ngate=256,
                  beamwidth=2.2,
                  )
    return altimeter(channel='Ku', **params)


# TODO: add Sentinel 6. Need to add non-circular footprints first.
