import pytest
import numpy as np

from smrt.core.error import SMRTError
from smrt.core.globalconstants import PERMITTIVITY_OF_FREE_SPACE
from smrt.substrate.flat import Flat
from smrt.inputs.make_bedrock import make_bedrock, BEDROCK_COMPLEX_PERMITTIVITY_DATA


def test_make_bedrock_constant_permittivity():
    bedrock = make_bedrock(substrate_model=Flat, bedrock_permittivity_model=3.14+0j, temperature=273.15)
    assert isinstance(bedrock, Flat)
    assert bedrock.permittivity(1e9) == 3.14+0j
    assert bedrock.temperature == 273.15

def test_make_bedrock_constant_permittivity_real():
    bedrock = make_bedrock(substrate_model=Flat, bedrock_permittivity_model=3.14, temperature=273.35)
    assert isinstance(bedrock, Flat)
    assert bedrock.permittivity(1e9) == 3.14
    assert bedrock.temperature == 273.35

def test_make_bedrock_named_model():
    bedrock = make_bedrock(
        substrate_model=Flat,
        bedrock_permittivity_model='granite_hartlieb16',
        temperature=273.15
    )
    assert isinstance(bedrock, Flat)
    # Test at a specific frequency, e.g., 2450 MHz as per the data comment
    assert bedrock.permittivity(2.45e9) == (5.45 + 0.038j)

def test_make_bedrock_named_model_with_conductivity():
    # Test a model with conductivity, where imaginary part should be 0 in the data
    bedrock = make_bedrock(
        substrate_model=Flat,
        bedrock_permittivity_model='frozen_bedrock_tulaczyk20',
        temperature=273.15
    )
    assert isinstance(bedrock, Flat)
    # For 'frozen_bedrock_tulaczyk20', complex_permittivity is 2.7+0j and conductivity is 0.0002
    # At 5e6 Hz
    expected_imaginary_part = 0.0002 / (2 * np.pi * 5e6 * PERMITTIVITY_OF_FREE_SPACE)
    assert bedrock.permittivity(5e6) == (2.7 + expected_imaginary_part * 1j)

def test_make_bedrock_missing_temperature():
    with pytest.raises(TypeError):
        make_bedrock(substrate_model=Flat, bedrock_permittivity_model=3.14) # type: ignore 

def test_make_bedrock_unrecognized_model_name():
    with pytest.raises(SMRTError):
        make_bedrock(substrate_model=Flat, bedrock_permittivity_model='non_existent_model', temperature=273.15)

def test_make_bedrock_callable_permittivity():
    def custom_permittivity(frequency: float, temperature: float) -> complex:
        return 5.0 + 0.1j * (frequency / 1e9)
    
    bedrock = make_bedrock(substrate_model=Flat, bedrock_permittivity_model=custom_permittivity, temperature=273.15)
    assert isinstance(bedrock, Flat)
    assert bedrock.permittivity(1e9) == (5.0 + 0.1j)
    assert bedrock.permittivity(2e9) == (5.0 + 0.2j)


# Test data table sanity
def test_bedrock_data_consistency():
    for name, (complex_permittivity, conductivity) in BEDROCK_COMPLEX_PERMITTIVITY_DATA.items():
        # If conductivity is non-zero, the imaginary part of complex_permittivity must be zero !
        if conductivity != 0:
            assert complex_permittivity.imag == 0, f"Model '{name}': Conductivity is non-zero, but imaginary permittivity is also non-zero."
