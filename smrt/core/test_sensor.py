
from nose.tools import raises
from nose.tools import eq_

from smrt.core import sensor
from smrt.core.error import SMRTError

# Generic test - store for later
# class FooTests(unittest.TestCase):
#
#     def testFoo(self):
#         self.failUnless(False)

# setup has not been used as various inputs will be used in the test


# passive test


@raises(SMRTError)
def test_passive_wrong_frequency_units_warning():
    sensor.passive([1e9, 35], theta=55)


def test_passive_mode():
    se = sensor.passive(35e9, 55, polarization="H")
    print(se.mode)


# active test


@raises(SMRTError)
def test_active_wrong_frequency_units_warning():
    sensor.active([1e9, 35], 55)


#def test_active_fourpol():
#    sensor = sensor.active(35e9, 55, polarization="4P")
#    assert "HH" in sensor.polarization
#    assert "VV" in sensor.polarization
#    assert "HV" in sensor.polarization
#    assert "VH" in sensor.polarization


def test_active_mode():
    se = sensor.active(35e9, 55)
    assert se.mode == 'A'

