
from nose.tools import raises
from nose.tools import eq_

from smrt.inputs.sensor_list import amsre
from smrt.core.error import SMRTError

# Generic test - store for later
# class FooTests(unittest.TestCase):
#
#     def testFoo(self):
#         self.failUnless(False)

# setup has not been used as various inputs will be used in the test



# AMSRE test

@raises(SMRTError)
def test_amsre_channel_recognized():
    # This tests that system error is raised if the AMSR-E channel is not in the dictionary list
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

def test_theta_is_55():
    se = amsre(channel='37V')
    eq_(se.theta, 0.95993108859688125)
