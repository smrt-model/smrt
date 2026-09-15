import numpy as np
import pytest

from smrt.core.error import SMRTError
from smrt.core.sensor import Altimeter, SensorList
from smrt.inputs.altimeter_list import (
    asiras_lam,
    cryosat2,
    envisat_ra2,
    sentinel3_sral,
)


def test_cryosat2_returns_configured_altimeter():
    sensor = cryosat2(pitch_angle_deg=2, roll_angle_deg=3)

    assert isinstance(sensor, Altimeter)
    assert sensor.channel_map == {"Ku": {}}
    assert sensor.frequency == 13.575e9
    assert sensor.altitude == 717_242
    assert sensor.ndoppler == 64
    assert sensor.doppler_window == "hamming"
    assert sensor.pitch_angle == pytest.approx(np.deg2rad(2))
    assert sensor.roll_angle == pytest.approx(np.deg2rad(3))


@pytest.mark.parametrize(
    "band, surface, frequency, doppler_window",
    [
        ("Ku", "ocean", 13.575e9, "hamming"),
        ("C", "landice", 5.41e9, "rect"),
        ("C", "seaice", 5.41e9, "hamming"),
    ],
)
def test_sentinel3_sral_maps_band_and_surface(band, surface, frequency, doppler_window):
    sensor = sentinel3_sral(band, surface)

    assert sensor.channel_map == {band: {}}
    assert sensor.frequency == frequency
    assert sensor.doppler_window == doppler_window


@pytest.mark.parametrize(
    "band, surface",
    [("X", "ocean"), ("Ku", "lake")],
)
def test_sentinel3_sral_rejects_unknown_configuration(band, surface):
    with pytest.raises(SMRTError):
        sentinel3_sral(band, surface)


def test_envisat_ra2_supports_single_and_all_channels():
    single_channel = envisat_ra2(channel="S")
    all_channels = envisat_ra2()

    assert isinstance(single_channel, Altimeter)
    assert list(single_channel.channel_map) == ["S"]
    assert single_channel.frequency == 3.2e9
    assert isinstance(all_channels, SensorList)
    assert all_channels.channel == ["Ku", "S"]
    assert all_channels.frequency == [13.575e9, 3.2e9]


def test_asiras_lam_requires_aircraft_altitude():
    with pytest.raises(SMRTError):
        asiras_lam()

    sensor = asiras_lam(altitude=5_000)
    assert sensor.altitude == 5_000
    assert sensor.beamwidth_along_track == 2.2
    assert sensor.beamwidth_cross_track == 9.8
