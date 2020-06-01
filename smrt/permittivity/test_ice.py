
import pytest

import warnings
import numpy as np

from smrt.permittivity.ice import ice_permittivity_maetzler06, \
                                    ice_permittivity_maetzler87,\
                                    ice_permittivity_tiuri84,\
                                    _ice_permittivity_HUT,\
                                    _ice_permittivity_DMRTML,\
                                    _ice_permittivity_MEMLS

from smrt.core.error import SMRTError

# Input temperature array functionality removed. If ever needed, use numpy instead of math in ice.py, but slower.

# @raises(SMRTError)
# def test_zero_temperature_exception_raised():
#    ice_permittivity_maetzler06(10e9, np.array([0]), 0, 0)


# This tests a warning is raised
# with warnings.catch_warnings(record=True) as w:
#     # Cause all warnings to always be triggered.
#     warnings.simplefilter("always")
#     # Trigger a warning.
#     ice_permittivity(np.array([230]), 10e9)
#     # Verify some things
#     assert len(w) == 1
#     assert 'Warning: temperature is below 240K. Ice permittivity is out of range of applicability' in str(w[-1].message)

# Test output of this module against output from MEMLS code
# Not exact as MEMLS references to 273, not 273.15
def test_ice_permittivity_output_matzler_temp_270():
    eps = ice_permittivity_maetzler06(10e9, 270)
    print(eps)
    np.testing.assert_allclose(eps.real, 3.18567, atol=1e-3)
    np.testing.assert_allclose(eps.imag, 9.093e-04, atol=1e-4)

# Test output of this module against output from MEMLS code
# Weaker tolerance for 250K as MEMLS calculation is based on freezing point temperature of 273K not 273.15K
def test_ice_permittivity_output_matzler_temp_250():
    eps = ice_permittivity_maetzler06(10e9, 250)
    print(eps)
    np.testing.assert_allclose(eps.real, 3.1675, atol=1e-3)
    np.testing.assert_allclose(eps.imag, 6.0571e-4, atol=1e-4)

def test_imaginary_ice_permittivity_output_matzler_temp_270_freq_20GHz():
    eps = ice_permittivity_maetzler06(20e9, 270)
    np.testing.assert_allclose(eps.imag, 0.0017449, atol=1e-4)


def test_imaginary_ice_permittivity_output_matzler_temp_250_freq_20GHz():
    eps = ice_permittivity_maetzler06(20e9, 250)
    np.testing.assert_allclose(eps.imag, 0.0012002, atol=1e-4)


def test_imaginary_ice_permittivity_output_matzler_temp_270_freq_30GHz():
    eps = ice_permittivity_maetzler06(30e9, 270)
    np.testing.assert_allclose(eps.imag, 0.0025971, atol=1e-4)


def test_imaginary_ice_permittivity_output_matzler_temp_250_freq_30GHz():
    eps = ice_permittivity_maetzler06(30e9, 250)
    np.testing.assert_allclose(eps.imag, 0.0017973, atol=1e-4)


def test_imaginary_ice_permittivity_output_matzler_temp_270_freq_40GHz():
    eps = ice_permittivity_maetzler06(40e9, 270)
    np.testing.assert_allclose(eps.imag, 0.0034535, atol=1e-4)


def test_imaginary_ice_permittivity_output_matzler_temp_250_freq_40GHz():
    eps = ice_permittivity_maetzler06(40e9, 250)
    np.testing.assert_allclose(eps.imag, 0.0023952, atol=1e-4)


# Test output of this maetzler 87 against output maetzler 06
def test_real_ice_permittivity_output_maetzler87_temp_268():
    eps87 = ice_permittivity_maetzler87(10e9, 268.15)
    eps06 = ice_permittivity_maetzler06(10e9, 268.15)
    assert eps87.real == eps06.real


# Test output of this maetzler 87 against manually calculated value
def test_imag_ice_permittivity_output_maetzler87_temp_minus5():
    eps = ice_permittivity_maetzler87(10e9, 268.15)
    np.testing.assert_allclose(eps.imag, 8.2368e-4, atol=1e-8)


# Test output of this maetzler 87 against manually calculated value
def test_imag_ice_permittivity_output_maetzler87_temp_minus15():
    eps = ice_permittivity_maetzler87(10e9, 258.15)
    np.testing.assert_allclose(eps.imag, 6.0556e-4, atol=1e-8)


# Test output of tuiri84 against manually calculated value
def test_ice_permittivity_output_tuiri84_temp_minus10_freq_10GHz():
    eps = ice_permittivity_tiuri84(10e9, 263.15)
    print(eps)
    np.testing.assert_allclose(eps.real, 3.1466272230000003, atol=1e-8)
    np.testing.assert_allclose(eps.imag, 0.0014719740743411925, atol=1e-8)


# Test output of tuiri84 against manually calculated value
def test_ice_permittivity_output_tuiri84_temp_minus10_freq_40GHz():
    eps = ice_permittivity_tiuri84(40e9, 263.15)
    print(eps)
    np.testing.assert_allclose(eps.real, 3.1466272230000003, atol=1e-8)
    np.testing.assert_allclose(eps.imag, 0.0027502673494269655, atol=1e-8)


# Test output of tuiri84 against manually calculated value
def test_ice_permittivity_output_tuiri84_temp_250K_freq_10GHz():
    eps = ice_permittivity_tiuri84(10e9, 250.15)
    print(eps)
    np.testing.assert_allclose(eps.real, 3.1466272230000003, atol=1e-8)
    np.testing.assert_allclose(eps.imag, 0.0009218289507887421, atol=1e-8)


# Test output of tuiri84 against manually calculated value
def test_ice_permittivity_output_tuiri84_temp_250K_freq_40GHz():
    eps = ice_permittivity_tiuri84(40e9, 263)
    print(eps)
    np.testing.assert_allclose(eps.real, 3.1466272230000003, atol=1e-8)
    np.testing.assert_allclose(eps.imag, 0.0027354559325573355, atol=1e-8)

# Test output of HUT version
def test_real_ice_permittivity_output_HUT():
    eps = _ice_permittivity_HUT(10e9, 270)
    assert np.allclose(eps.real, 3.18567)


# Test output of HUT version
def test_imaginary_ice_permittivity_output_HUT():
    eps = _ice_permittivity_HUT(10e9, 270)
    np.testing.assert_allclose(eps.imag, 8.86909246416410e-04, atol=1e-8)


# Test output of DMRT version
def test_real_ice_permittivity_output_DMRTML():
    eps = _ice_permittivity_DMRTML(10e9, 270)
    assert np.allclose(eps.real, 3.18567)


# Test output of DMRT version
def test_imaginary_ice_permittivity_output_DMRTML():
    eps = _ice_permittivity_DMRTML(10e9, 270)
    np.testing.assert_allclose(eps.imag, 9.0679820556720323e-04, atol=1e-8)


# Test output MEMLS version
# Should be exact
def test_real_ice_permittivity_output_matzler_temp_270_MEMLS():
    eps = _ice_permittivity_MEMLS(10e9, 270, 0)
    assert np.allclose(eps.real, 3.18567)


# Test output MEMLS version
def test_imaginary_ice_permittivity_output_matzler_temp_270_freq_10GHz():
    eps = _ice_permittivity_MEMLS(10e9, 270, 0)
    np.testing.assert_allclose(eps.imag, 9.09298888985990e-04, atol=1e-8)


# Test output MEMLS version
def test_salty_imaginary_ice_permittivity_output_matzler_temp_270_freq_10GHz():
    eps = _ice_permittivity_MEMLS(10e9, 270, 50)
    np.testing.assert_allclose(eps.imag, 7.74334595964606, atol=1e-8)
