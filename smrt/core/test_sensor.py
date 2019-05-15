
from nose.tools import raises
from nose.tools import eq_

import numpy as np

from smrt.core import sensor
from smrt.core.error import SMRTError

# Generic test - store for later
# class FooTests(unittest.TestCase):
#
#     def testFoo(self):
#         self.failUnless(False)

# setup has not been used as various inputs will be used in the test


# passive test


def test_sensor_list():
    s = sensor.SensorList([sensor.active(f, 55, channel="%i" % (f/1e9)) for f in [1e9, 2e9, 3e9]])
    assert len(list(s.iterate())) == 3


def test_iterate():
    freqs = [1e9, 2e9, 3e9]
    s = sensor.active(freqs, 55)
    
    freqs_bis = [sub_s.frequency for sub_s in s. iterate("frequency")]

    np.testing.assert_equal(freqs, freqs_bis)


def test_wavelength():
    s = sensor.Sensor(wavelength=0.21, theta_deg=0)
    assert s.wavelength==0.21
    assert np.allclose(s.frequency, 1427583133)


@raises(SMRTError)
def test_no_theta():
    sensor.passive(1e9, theta=None)


@raises(SMRTError)
def test_passive_wrong_frequency_units_warning():
    sensor.passive([1e9, 35], theta=55)


@raises(SMRTError)
def test_duplicate_theta():
    sensor.passive([1e9, 35], theta=[55, 55])


@raises(SMRTError)
def test_duplicate_theta_active():
    sensor.active([1e9, 35], [55, 55])


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

