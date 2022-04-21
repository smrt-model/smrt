
import pytest

import numpy as np

from smrt.core import sensor
from smrt.core.error import SMRTError, SMRTWarning

# Generic test - store for later
# class FooTests(unittest.TestCase):
#
#     def testFoo(self):
#         self.failUnless(False)

# setup has not been used as various inputs will be used in the test


# passive test


def test_iterate():
    freqs = [1e9, 2e9, 3e9]
    s = sensor.active(freqs, 55)

    freqs_bis = [sub_s.frequency for sub_s in s. iterate("frequency")]

    np.testing.assert_equal(freqs, freqs_bis)


def test_wavelength():
    s = sensor.Sensor(wavelength=0.21, theta_deg=0)
    assert s.wavelength == 0.21
    assert np.allclose(s.frequency, 1427583133)


def test_no_theta():
    with pytest.raises(SMRTError):
        sensor.passive(1e9, theta=None)


def test_passive_wrong_frequency_units_warning():
    with pytest.warns(SMRTWarning):
        sensor.passive([1e9, 35], theta=55)


def test_duplicate_theta():
    with pytest.raises(SMRTError):
        sensor.passive([1e9, 35], theta=[55, 55])


def test_duplicate_theta_active():
    with pytest.raises(SMRTError):
        sensor.active([1e9, 35], [55, 55])


def test_passive_mode():
    se = sensor.passive(35e9, 55, polarization="H")
    print(se.mode)


# active test


def test_active_wrong_frequency_units_warning():
    with pytest.warns(SMRTWarning):
        sensor.active([1e9, 35], 55)


# def test_active_fourpol():
#    sensor = sensor.active(35e9, 55, polarization="4P")
#    assert "HH" in sensor.polarization
#    assert "VV" in sensor.polarization
#    assert "HV" in sensor.polarization
#    assert "VH" in sensor.polarization


def test_active_mode():
    se = sensor.active(35e9, 55)
    assert se.mode == 'A'
