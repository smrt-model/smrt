
import numpy as np
import pytest

from smrt.core.globalconstants import FREEZING_POINT
from smrt.inputs.make_soil import make_soil
from smrt.permittivity.water import water_permittivity


def test_make_rough_choudhury():

    make_soil('rough_choudhury79', 'dobson85', 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100, roughness_rms=1e-2)


def test_rough_choudhury_reflection():

    s = make_soil('rough_choudhury79', 'dobson85', 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100, roughness_rms=1e-3)

    frequency = 1e9
    mu1 = np.cos(np.radians(np.arange(10, 80)))

    npol = 2

    s.specular_reflection_matrix(frequency, 1, mu1, npol)


def test_raises_ksigma_warning():
    with pytest.raises(Warning):
        s = make_soil('rough_choudhury79', 'dobson85', 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100, roughness_rms=0.1)
        frequency = 1e9
        mu1 = np.cos(np.radians(np.arange(10, 80)))
        npol = 2
        s.specular_reflection_matrix(frequency, 1, mu1, npol)


def test_make_rough_water():
    make_soil('rough_choudhury79', water_permittivity(6e9, FREEZING_POINT), temperature=270, roughness_rms=0.5e-3)


def test_equivalence_fresnel():

    h2o = water_permittivity(6e9, FREEZING_POINT)
    rough = make_soil('rough_choudhury79', h2o, temperature=FREEZING_POINT, roughness_rms=0.)
    smooth = make_soil('flat', h2o, temperature=FREEZING_POINT)
    np.testing.assert_equal(rough.specular_reflection_matrix(10e9, 1, np.cos(50), 2)[0], smooth.specular_reflection_matrix(10e9, 1, np.cos(50), 2)[0])
