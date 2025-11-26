import pytest

from smrt.core.error import SMRTError
from smrt.inputs.sensor_list import amsr2, amsre, cimr

# Generic test - store for later
# class FooTests(unittest.TestCase):
#
#     def testFoo(self):
#         self.failUnless(False)


@pytest.mark.parametrize("sensor", [(amsre), (amsr2)])
def test_amsr_channel_recognized(sensor):
    # This tests that system error is raised if the AMSR-E channel is not in the dictionary list
    with pytest.raises(SMRTError):
        sensor(channel="20H")


@pytest.mark.parametrize(
    "sensor, channel, frequency, polarization",
    [
        (amsre, "19H", 18.7e9, ["H"]),
        (amsre, "37V", 36.5e9, ["V"]),
        (amsr2, "06H", 6.925e9, ["H"]),
        (amsr2, "07V", 7.3e9, ["V"]),
        (cimr, "01H", 1.4135e9, ["H"]),
    ],
)
def test_map_channel_to_dictionary(sensor, channel, frequency, polarization):
    # Test to ensure channel frequency of 19H is assigned 18.7GHz frequency and polarization H
    se = sensor(channel=channel)
    assert se.frequency == frequency
    assert se.polarization == polarization


@pytest.mark.parametrize("sensor", [(amsre), (amsr2), (cimr)])
def test_amsr_theta_is_55(sensor):
    se = sensor(channel="37V")
    assert se.theta == 0.95993108859688125
