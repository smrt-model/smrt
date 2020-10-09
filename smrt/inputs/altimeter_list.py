

from smrt.core.sensor import make_multi_channel_altimeter, altimeter

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
