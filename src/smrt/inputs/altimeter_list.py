"""This module contains a list of altimeter sensor configurations for various satellite missions.
The configurations are defined as functions that return an Altimeter instance with the appropriate parameters.
The functions are named according to the satellite mission and sometimes the altimeter type (LRM or SAR).
"""

from warnings import warn

from smrt.core.error import SensorList, SMRTError
from smrt.core.sensor import Altimeter


def altimeter(channel, **kwargs):
    """
    Return a generic configuration for a SAR or LRM altimeter.

    """

    return Altimeter(channel=channel, **kwargs)


def make_multi_channel_altimeter(config, channel):
    # helper function to make a single or multi channel altimeter sensor object from a config in dict format
    if isinstance(channel, str):
        return Altimeter(channel=channel, **config[channel])
    else:
        if channel is None:
            channel = config.keys()
        return SensorList([Altimeter(channel=c, **config[c]) for c in channel])


#
# List of SAR altimeters
#


def cryosat2(pitch_angle_deg=0, roll_angle_deg=0, force_circular_antenna=False):
    """Return an altimeter instance for CryoSat-2: SAR mode without oversampling but with hamming window

    Parameters from https://docslib.org/doc/3527464/cryosat-product-handbook
    https://earth.esa.int/eogateway/documents/20142/37627/CryoSat-Baseline-D-Product-Handbook.pdf  (page 13)
    """

    params = dict(
        frequency=13.575e9,  # in Hz
        altitude=717_242,  # in m
        pulse_bandwidth=320e6,  # in Hz
        pulse_repetition_frequency=17825,  # in Hz
        # FYI pulse_duration=44.8e-6,  # in s
        velocity=7435,  # m/s
        ngate=128,  # here we consider the initial configuration of the SAR Mode. No oversampling
        ndoppler=64,
        nominal_gate=44,  # Estimate - needs better definition
        beamwidth_alongtrack=1.08,
        beamwidth_acrosstrack=1.2,
        doppler_window="hamming",
        antenna_gain=1,  # one-way antenna gain
    )
    if force_circular_antenna:
        beamwidth = (params["beamwidth_alongtrack"] + params["beamwidth_acrosstrack"]) / 2
        params["beamwidth_alongtrack"] = beamwidth
        params["beamwidth_acrosstrack"] = beamwidth

    return altimeter(channel="Ku", **params, pitch_angle_deg=pitch_angle_deg, roll_angle_deg=roll_angle_deg)


def cryosat2_sarm(*args, **kwargs):
    warn(
        "This function is deprecated and will be removed in a future version. "
        "Use cryosat2 instead but be aware that the altitude and nominal gate are slightly different.",
        DeprecationWarning,
    )
    cryosat2(*args, **kwargs)


def sentinel3_sral(band: str, surface: str, pitch_angle_deg=0, roll_angle_deg=0):
    """Return an altimeter instance for Sentienl 3 for both SAR and LRM modes.

    Documented in: https://sentinel.esa.int/documents/247904/4871083/Sentinel-3+SRAL+Land+User+Handbook+V1.1.pdf
    """

    if surface == "landice":
        doppler_window = "rect"
    elif surface in ["ocean", "seaice"]:
        doppler_window = "hamming"
    else:
        raise SMRTError("Invalid surface. Must be landice, ocean or seaice.")

    if band == "Ku":
        params = dict(
            frequency=13.575e9,  # in Hz
            altitude=814_500,  # in m
            pulse_bandwidth=320e6,  # in Hz
            pulse_repetition_frequency=17825,  # in Hz
            # FYI pulse_duration=48.95e-6,  # in s
            velocity=7450,  # m/s
            ngate=128,  # here we consider the initial configuration of the SAR Mode. No oversampling
            ndoppler=64,
            nominal_gate=44,  # Estimate - needs better definition
            beamwidth_alongtrack=1.35,
            beamwidth_acrosstrack=1.35,
            doppler_window="hamming",
            antenna_gain=1,  # one-way antenna gain
        )
    elif band == "C":
        params = dict(
            frequency=5.41e9,  # in Hz
            altitude=814_500,  # in m
            pulse_bandwidth=290e6,  # in Hz
            pulse_repetition_frequency=2 * 78.5,  # two pulse every ~80 Hz
            # FYI pulse_duration=48.95e-6,  # in s
            velocity=7450,  # m/s
            ngate=128,  # here we consider the initial configuration of the SAR Mode. No oversampling
            ndoppler=2,
            nominal_gate=44,  # Estimate - needs better definition
            beamwidth_alongtrack=3.4,  # estimated proportionnaly to Ku charcateristics...
            beamwidth_acrosstrack=3.4,
            antenna_gain=1,  # one-way antenna gain
            doppler_window=doppler_window,
        )
    else:
        raise SMRTError("Invalid band. Must be Ku or C.")

    return altimeter(channel=band, **params, pitch_angle_deg=pitch_angle_deg, roll_angle_deg=roll_angle_deg)


def sentinel3_sarm(*args, **kwargs):
    warn(
        "This function is deprecated and will be removed in a future version. Use sentinel3_ra2 instead",
        DeprecationWarning,
    )
    sentinel3_sral(*args, **kwargs)


def cristal(band: str, pitch_angle_deg=0, roll_angle_deg=0, force_circular_antenna=False):
    """Return an altimeter instance for CryoSat-2: SAR mode without oversampling but with hamming window

    Parameters from https://docslib.org/doc/3527464/cryosat-product-handbook
    https://earth.esa.int/eogateway/documents/20142/37627/CryoSat-Baseline-D-Product-Handbook.pdf  (page 13)
    """

    if band == "Ku":
        params = dict(
            frequency=13.575e9,  # in Hz
            altitude=699_000,  # in m
            pulse_bandwidth=500e6,  # in Hz
            pulse_repetition_frequency=17825,  # in Hz  # from Cryosat2
            velocity=7524,  # m/s
            ngate=256,  # here we consider the initial configuration of the SAR Mode. No oversampling
            ndoppler=128,  # first guess
            nominal_gate=44,  # Estimate - needs better definition
            beamwidth_alongtrack=1.08,  # from Cryosat2
            beamwidth_acrosstrack=1.2,  # from Cryosat2
            doppler_window="hamming",
            antenna_gain=1,  # one-way antenna gain
        )
    elif band == "Ka":
        params = dict(
            frequency=35.75e9,  # in Hz
            altitude=699_000,  # in m
            pulse_bandwidth=500e6,  # in Hz
            pulse_repetition_frequency=17825,  # in Hz
            velocity=7524,  # m/s
            ngate=256,  # here we consider the initial configuration of the SAR Mode. No oversampling
            ndoppler=64,
            nominal_gate=44,  # Estimate - needs better definition
            beamwidth_alongtrack=1.08 * 13.5 / 35.7,  # from Cryosat2 and scaled by the frequency
            beamwidth_acrosstrack=1.2 * 13.5 / 35.7,
            doppler_window="hamming",
            antenna_gain=1,  # one-way antenna gain
        )
    else:
        raise SMRTError("Invalid band. Must be Ku or Ka.")

    if force_circular_antenna:
        beamwidth = (params["beamwidth_alongtrack"] + params["beamwidth_acrosstrack"]) / 2
        params["beamwidth_alongtrack"] = beamwidth
        params["beamwidth_acrosstrack"] = beamwidth

    return altimeter(channel="Ku", **params, pitch_angle_deg=pitch_angle_deg, roll_angle_deg=roll_angle_deg)


#
# List of specific LRM altimeters
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
    return altimeter(channel="Ka", **params)


def asiras_lam(altitude=None, pitch_angle_deg=0, roll_angle_deg=0):
    """Return an altimeter instance for ASIRAS in Low Altitude Mode

    Parameters from https://earth.esa.int/web/eoportal/airborne-sensors/asiras
    Beam width is 2.2 x 9.8 deg

    Brown1997 can not take elliptical footprints into account whereas Newkirk1992 model can. Select it in the waveform
    model.
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
    return altimeter(channel="Ku", **params)


def cryosat2_lrm(pitch_angle_deg=0, roll_angle_deg=0):
    """Return an altimeter instance for CryoSat-2 in LRM model

    Parameters from https://earth.esa.int/web/eoportal/satellite-missions/c-missions/cryosat-2
    Altitude from https://doi.org/10.1016/j.asr.2018.04.014
    Beam width is 1.08 along track and 1.2 across track

    """

    warn(
        "This functon is deprecated and will be removed in a future version. Use cryosat2 instead but be aware that the"
        "altitude and nominal gate are slightly different.",
        DeprecationWarning,
    )

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
    return altimeter(channel="Ku", **params)
