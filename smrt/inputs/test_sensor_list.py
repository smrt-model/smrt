
import pytest

from smrt.inputs.sensor_list import amsre, amsr2, cimr
from smrt.core.error import SMRTError

# Generic test - store for later
# class FooTests(unittest.TestCase):
#
#     def testFoo(self):
#         self.failUnless(False)

# setup has not been used as various inputs will be used in the test


# AMSRE test

def test_amsre_channel_recognized():
    # This tests that system error is raised if the AMSR-E channel is not in the dictionary list
    with pytest.raises(SMRTError):
        se = amsre(channel='20H')


def test_map_channel19_to_dictionary():
    # Test to ensure channel frequency of 19H is assigned 18.7GHz frequency and polarization H
    se = amsre(channel='19H')
    assert se.frequency == 18.7e9
    assert se.polarization == ['H']


def test_map_channel37_to_dictionary():
    # Test to ensure channel frequency of 37V is assigned 36.5GHz frequency and polarization V
    se = amsre(channel='37V')
    assert se.frequency == 36.5e9
    assert se.polarization == ['V']


def test_amsre_theta_is_55():
    se = amsre(channel='37V')
    assert se.theta == 0.95993108859688125

# AMSR2 test


def test_amsre_channel_recognized():
    # This tests that system error is raised if the AMSR-E channel is not in the dictionary list
    with pytest.raises(SMRTError):
        se = amsr2(channel='20H')


def test_map_channel06_to_dictionary():
    # Test to ensure channel frequency of 06H is assigned 6 GHz frequency and polarization H
    se = amsr2(channel='06H')
    assert se.frequency == 6.925e9
    assert se.polarization == ['H']


def test_map_channel07_to_dictionary():
    # Test to ensure channel frequency of 07V is assigned 7.3 GHz frequency and polarization V
    se = amsr2(channel='07V')
    assert se.frequency == 7.3e9
    assert se.polarization == ['V']


def test_amsr2_theta_is_55():
    se = amsr2(channel='37V')
    assert se.theta == 0.95993108859688125

# CIMR test

def test_cimr_channel01_to_dictionary():
    # Test to ensure channel frequency of 19H is assigned 18.7GHz frequency and polarization H
    se = cimr(channel='01H')
    assert se.frequency == 1.4135e9
    assert se.polarization == ['H']


def test_cimr_is_55():
    se = cimr(channel='37V')
    assert se.theta == 0.95993108859688125
