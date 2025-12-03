from smrt.core.error import SMRTError
from smrt.core.sensor import sar_altimeter


def cryosat2_sarm(pitch_angle_deg=0, roll_angle_deg=0, force_circular_antenna=False):
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

    return sar_altimeter(channel="Ku", **params, pitch_angle_deg=pitch_angle_deg, roll_angle_deg=roll_angle_deg)


def sentinel3_sarm(band: str, surface: str, pitch_angle_deg=0, roll_angle_deg=0):
    """Return an altimeter instance for Sentienl 3

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

    return sar_altimeter(channel=band, **params, pitch_angle_deg=pitch_angle_deg, roll_angle_deg=roll_angle_deg)


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

    return sar_altimeter(channel="Ku", **params, pitch_angle_deg=pitch_angle_deg, roll_angle_deg=roll_angle_deg)
