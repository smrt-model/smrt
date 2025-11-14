import numpy as np
import pytest
from smrt.permittivity.ice import (
    _ice_permittivity_DMRTML,
    _ice_permittivity_HUT,
    _ice_permittivity_MEMLS,
    ice_permittivity_hufford91_maetzler87,
    ice_permittivity_maetzler06,
    ice_permittivity_maetzler87,
    ice_permittivity_tiuri84,
)

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
@pytest.mark.parametrize("permittivity_model, frequency, temperature, real, imag, ratol, iatol",
                         [(ice_permittivity_maetzler06, 10e9, 270, 3.18567, 9.093e-04, 1e-3, 1e-4),                              (ice_permittivity_maetzler06, 10e9, 250, 3.1675, 6.0571e-4, 1e-3, 1e-4),
                          (ice_permittivity_tiuri84, 10e9, 263.15, 3.1466272230000003, 0.0014719740743411925, 1e-8, 1e-8),
                          (ice_permittivity_tiuri84, 40e9, 263.15, 3.1466272230000003, 0.0027502673494269655, 1e-8, 1e-8),
                          (ice_permittivity_tiuri84, 10e9, 250.15, 3.1466272230000003, 0.0009218289507887421, 1e-8, 1e-8),
                          (_ice_permittivity_HUT, 10e9, 270, 3.18567, 8.86909246416410e-04, 1e-8, 1e-8),
                          (_ice_permittivity_DMRTML, 10e9, 270, 3.18567, 9.0679820556720323e-04, 1e-8, 1e-8),
                          (ice_permittivity_hufford91_maetzler87, 10e9, 270, 3.18567, 0.0009650945, 1e-8, 1e-8)])
def test_ice_permittivity(permittivity_model, frequency, temperature, real, imag, ratol, iatol):
    eps = permittivity_model(frequency, temperature)
    print(eps)
    np.testing.assert_allclose(eps.real, real, atol=ratol)
    np.testing.assert_allclose(eps.imag, imag, atol=iatol)

@pytest.mark.parametrize("permittivity_model, frequency, temperature, exp_eps_imag, atol",
                         [(ice_permittivity_maetzler06, 20e9, 270, 0.0017449, 1e-4),
                                                                  (ice_permittivity_maetzler06, 20e9, 250, 0.0012002, 1e-4),
                                                                  (ice_permittivity_maetzler06, 30e9, 270, 0.0025971, 1e-4),
                                                                  (ice_permittivity_maetzler06, 30e9, 250, 0.0017973, 1e-4),
                                                                  (ice_permittivity_maetzler06, 40e9, 270, 0.0034535, 1e-4),
                                                                  (ice_permittivity_maetzler06, 40e9, 250, 0.0023952, 1e-4),
                                                                  (ice_permittivity_maetzler87, 10e9, 268.15, 8.2368e-4, 1e-8),
                                                                  (ice_permittivity_maetzler87, 10e9, 258.15, 6.0556e-4, 1e-8)])
def test_imaginary_ice_permittivity_output_matzler(permittivity_model,frequency, temperature, exp_eps_imag, atol):
    eps = permittivity_model(frequency=frequency, temperature=temperature)
    np.testing.assert_allclose(eps.imag, exp_eps_imag, atol=atol)

# Test output of this maetzler 87 against output maetzler 06
def test_real_ice_permittivity_output_maetzler87_temp_268():
    eps87 = ice_permittivity_maetzler87(10e9, 268.15)
    eps06 = ice_permittivity_maetzler06(10e9, 268.15)
    assert eps87.real == eps06.real

# Test output MEMLS version
# Should be exact
def test_ice_permittivity_output_matzler_temp_270_MEMLS():
    eps = _ice_permittivity_MEMLS(10e9, 270, 0)
    assert np.allclose(eps.real, 3.18567)
    np.testing.assert_allclose(eps.imag, 9.09298888985990e-04, atol=1e-8)

# Test output MEMLS version
def test_salty_imaginary_ice_permittivity_output_matzler_temp_270_freq_10GHz():
    eps = _ice_permittivity_MEMLS(10e9, 270, 50)
    np.testing.assert_allclose(eps.imag, 7.74334595964606, atol=1e-8)
