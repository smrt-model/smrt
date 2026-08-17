from smrt.core.error import SMRTError
from smrt.core.sensor import lrm_altimeter, make_multi_channel_altimeter

#
# List of LRM altimeters
#


def envisat_ra2(channel=None, pitch_angle_deg=0, roll_angle_deg=0):
    """
    Returns an Altimeter instance for the ENVISAT RA2 altimeter.

    Args:
      channel: can be 'S', 'Ku', or both. Default is both.
    """

    config = {
        "Ku": dict(
            frequency=13.575e9,
            altitude=800e3,
            pulse_bandwidth=320e6,
            ngate=128,
            nominal_gate=45,
            beamwidth_alongtrack=1.29,
            beamwidth_acrosstrack=1.29,
            pitch_angle_deg=pitch_angle_deg,
            roll_angle_deg=roll_angle_deg,
        ),
        "S": dict(
            frequency=3.2e9,
            altitude=800e3,
            pulse_bandwidth=160e6,
            ngate=128,
            nominal_gate=32,  # to correct, the value is rather close to 25
            beamwidth_alongtrack=5.5,  # Lacroix et al. and Fatras et al.,
            beamwidth_acrosstrack=5.5,
            pitch_angle_deg=pitch_angle_deg,
            roll_angle_deg=roll_angle_deg,
        ),
    }

    return make_multi_channel_altimeter(config, channel)


def sentinel3_sral(channel=None, pitch_angle_deg=0, roll_angle_deg=0):
    """return an Altimeter instance for the Sentinel 3 SRAL instrument.

    :param channel: can be 'Ku' only ('C' is to be implemented)

    """

    config = {
        "Ku": dict(
            frequency=13.575e9,
            altitude=814e3,
            pulse_bandwidth=320e6,
            nominal_gate=44,
            ngate=128,
            beamwidth_alongtrack=1.35,
            beamwidth_acrosstrack=1.35,
            antenna_gain=1,
            pitch_angle_deg=pitch_angle_deg,
            roll_angle_deg=roll_angle_deg,
        ),
    }

    return make_multi_channel_altimeter(config, channel)


def saral_altika(pitch_angle_deg=0, roll_angle_deg=0):
    """return an Altimeter instance for the Saral/AltiKa instrument."""

    params = dict(
        frequency=35.75e9,
        altitude=800e3,
        pulse_bandwidth=480e6,
        nominal_gate=51,
        ngate=128,
        beamwidth_alongtrack=0.605,
        beamwidth_acrosstrack=0.605,
        antenna_gain=1,
        pitch_angle_deg=pitch_angle_deg,
        roll_angle_deg=roll_angle_deg,
    )
    return lrm_altimeter(channel="Ka", **params)


def cryosat2_lrm(pitch_angle_deg=0, roll_angle_deg=0):
    """Return an altimeter instance for CryoSat-2

    Parameters from https://earth.esa.int/web/eoportal/satellite-missions/c-missions/cryosat-2
    Altitude from https://doi.org/10.1016/j.asr.2018.04.014
    Beam width is 1.08 along track and 1.2 across track

    """

    params = dict(
        frequency=13.575e9,
        altitude=720e3,
        pulse_bandwidth=320e6,
        nominal_gate=50,  # Estimate - needs better definition
        ngate=128,
        beamwidth_alongtrack=1.08,
        beamwidth_acrosstrack=1.2,
        antenna_gain=1,
        pitch_angle_deg=pitch_angle_deg,
        roll_angle_deg=roll_angle_deg,
    )
    return lrm_altimeter(channel="Ku", **params)


# def cryosat2_sin(pitch_angle_deg=0, roll_angle_deg=0):
#     """Return an altimeter instance for CryoSat-2: SIN mode

#     Parameters from https://earth.esa.int/web/eoportal/satellite-missions/c-missions/cryosat-2
#     Altitude from https://doi.org/10.1016/j.asr.2018.04.014
#     Beam width is 1.08 along track and 1.2 across track. Brown1977 is not able to take into
#     account elliptical footprints yet in LRM, however Newkirk1992 model can. Select it in the waveform model.

#     """

#     params = dict(
#         frequency=13.575e9,
#         altitude=720e3,
#         pulse_bandwidth=320e6,
#         nominal_gate=164,  # Estimate - needs better definition
#         ngate=512,
#         beamwidth_alongtrack=1.08,
#         beamwidth_acrosstrack=1.2,
#         antenna_gain=1,
#         pitch_angle_deg=pitch_angle_deg,
#         roll_angle_deg=roll_angle_deg,
#     )
#     return lrm_altimeter(channel="Ku", **params)


def asiras_lam(altitude=None, pitch_angle_deg=0, roll_angle_deg=0):
    """Return an altimeter instance for ASIRAS in Low Altitude Mode

    Parameters from https://earth.esa.int/web/eoportal/airborne-sensors/asiras
    Beam width is 2.2 x 9.8 deg

    Brown1997 can not take elliptical footprints into account whereas Newkirk1992 model can. Select it in the waveform model.
    """
    if altitude is None:
        raise SMRTError("Aircraft altitude must be defined")
    else:
        altitude = altitude

    params = dict(
        frequency=13.5e9,
        pulse_bandwidth=1e9,
        altitude=altitude,
        nominal_gate=41,  # Estimate - needs better definition
        ngate=256,
        beamwidth_alongtrack=2.2,
        beamwidth_acrosstrack=9.8,
        antenna_gain=1,
        pitch_angle_deg=pitch_angle_deg,
        roll_angle_deg=roll_angle_deg,
    )
    return lrm_altimeter(channel="Ku", **params)
