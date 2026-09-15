import numpy as np
import pytest

from smrt.core.error import SMRTError, SMRTWarning
from smrt.core.globalconstants import C_SPEED
from smrt.core.sensor import Sensor

# Generic test - store for later
# class FooTests(unittest.TestCase):
#
#     def testFoo(self):
#         self.failUnless(False)

# setup has not been used as various inputs will be used in the test


# passive test


def test_iterate():
    freqs = [1e9, 2e9, 3e9]
    s = Sensor(freqs, theta_inc_deg=55, theta_deg=55, phi_deg=180, polarization_inc=["V", "H"], polarization=["V", "H"])
    s.basic_checks()

    freqs_bis = [sub_s.frequency for sub_s in s.iterate("frequency")]

    np.testing.assert_equal(freqs, freqs_bis)


def test_iterate_wavelength():
    freqs = [1e9, 2e9, 3e9]
    s = Sensor(freqs, theta_inc_deg=55, theta_deg=55, phi_deg=180, polarization_inc=["V", "H"], polarization=["V", "H"])
    s.basic_checks()

    wavelengths_bis = [sub_s.wavelength for sub_s in s.iterate("frequency")]

    np.testing.assert_equal(C_SPEED / np.array(freqs), wavelengths_bis)


def test_wavelength():
    s = Sensor(wavelength=0.21, theta_deg=0)
    np.testing.assert_allclose(s.wavelength, 0.21)
    np.testing.assert_allclose(s.frequency, 1427583133)


def test_no_theta():
    with pytest.raises(SMRTError):
        Sensor(1e9, theta_deg=None)


def test_passive_wrong_frequency_units_warning():
    with pytest.warns(SMRTWarning):
        sensor = Sensor([1e9, 35], theta_deg=55, polarization=["V", "H"])
        sensor.basic_checks()


def test_duplicate_theta():
    with pytest.raises(SMRTError):
        Sensor([1e9, 35], theta_deg=[55, 55], polarization=["V", "H"])


def test_duplicate_theta_active():
    with pytest.raises(SMRTError):
        Sensor(
            [1e9, 35],
            theta_inc_deg=[55, 55],
            theta_deg=[55, 55],
            phi_deg=180,
            polarization_inc=["V", "H"],
            polarization=["V", "H"],
        )


def test_passive_mode():
    se = Sensor(35e9, theta_deg=55, polarization="H")
    se.basic_checks()
    print(se.mode)


# active test


def test_active_wrong_frequency_units_warning():
    with pytest.warns(SMRTWarning):
        sensor = Sensor(
            [1e9, 35],
            theta_inc_deg=55,
            theta_deg=55,
            phi_deg=180,
            polarization_inc=["V", "H"],
            polarization=["V", "H"],
        )
        sensor.basic_checks()


# def test_active_fourpol():
#    sensor = sensor.active(35e9, 55, polarization="4P")
#    assert "HH" in sensor.polarization
#    assert "VV" in sensor.polarization
#    assert "HV" in sensor.polarization
#    assert "VH" in sensor.polarization


def test_active_mode():
    se = Sensor(
        35e9,
        theta_inc_deg=55,
        theta_deg=55,
        phi_deg=180,
        polarization_inc=["V", "H"],
        polarization=["V", "H"],
    )
    se.basic_checks()
    assert se.mode == "A"
