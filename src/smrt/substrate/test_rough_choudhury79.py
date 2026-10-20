import numpy as np
import pytest

from smrt.core.error import SMRTWarning
from smrt.core.globalconstants import FREEZING_POINT
from smrt.inputs.make_soil import make_soil
from smrt.permittivity.water import water_permittivity


@pytest.fixture
def rough_choudhury():
    return make_soil(
        "rough_choudhury79",
        "soil_permittivity_dobson85_peplinski95",
        275,
        moisture=0.9,
        sand=0.2,
        clay=0.3,
        drymatter=1100,
        roughness_rms=0.4e-3,
    )


def test_make_rough_choudhury(rough_choudhury):
    assert rough_choudhury.roughness_rms == 0.4e-3


def test_rough_choudhury_reflection(rough_choudhury):
    s = rough_choudhury

    frequency = 1e9
    mu1 = np.cos(np.radians(np.arange(10, 80)))

    for npol in (2, 3):
        refl = s.specular_reflection_matrix(frequency, 1, mu1, npol)
        assert refl.values.shape == (npol, len(mu1))
        assert np.all(np.isfinite(refl.values))


def test_rough_choudhury_emissivity_and_unsupported_polarizations(rough_choudhury):
    s = rough_choudhury

    frequency = 1e9
    mu1 = np.cos(np.radians(np.arange(10, 80)))

    for npol in (2, 3):
        emissivity = s.coherent_transmission_matrix(frequency, 1, mu1, npol)
        assert emissivity.values.shape == (npol, len(mu1))
        assert np.all(np.isfinite(emissivity.values))

    with pytest.raises(NotImplementedError):
        s.specular_reflection_matrix(frequency, 1, mu1, 4)
    with pytest.raises(Exception, match="to be implemented"):
        s.coherent_transmission_matrix(frequency, 1, mu1, 4)


def test_warns_ksigma_outside_validity(rough_choudhury):
    s = rough_choudhury
    s.roughness_rms = 1e-3
    frequency = 10e9
    mu1 = 1
    npol = 2

    with pytest.warns(SMRTWarning, match="outside validity range"):
        s.specular_reflection_matrix(frequency, 1, mu1, npol)


def test_make_rough_water():
    make_soil("rough_choudhury79", water_permittivity(6e9, FREEZING_POINT), temperature=270, roughness_rms=0.5e-3)


def test_equivalence_fresnel():
    h2o = water_permittivity(6e9, FREEZING_POINT)
    rough = make_soil("rough_choudhury79", h2o, temperature=FREEZING_POINT, roughness_rms=0.0)
    smooth = make_soil("flat", h2o, temperature=FREEZING_POINT)
    np.testing.assert_equal(
        rough.specular_reflection_matrix(10e9, 1, np.cos(50), 2)[0],
        smooth.specular_reflection_matrix(10e9, 1, np.cos(50), 2)[0],
    )
