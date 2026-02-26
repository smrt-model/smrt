import numpy as np
import pytest

from smrt import PSU
from smrt.core.globalconstants import FREEZING_POINT
from smrt.permittivity.ice import ice_permittivity_maetzler06
from smrt.permittivity.saline_ice import (
    impure_ice_permittivity_maetzler06,
    saline_ice_permittivity_pvs_mixing,
)


# Test impure ice with zero salinity same as pure ice permittivity
@pytest.mark.parametrize("impure_model", [impure_ice_permittivity_maetzler06, saline_ice_permittivity_pvs_mixing])
def test_impure_permittivity_same_as_pure_for_zero_salinty(impure_model):
    pure = ice_permittivity_maetzler06(18e9, 265)
    impure = impure_model(18e9, 265, 0)
    assert abs(pure.real - impure.real) < 1e-4
    assert abs(pure.imag - impure.imag) < 1e-7


# Test of impure ice at freezing point temperature, 0.013PSU salinity, 10GHz
def test_impure_ice_freezing_point_0p013psu_10GHz():
    delta_epsimag = 1.0 / (1866 * np.exp(-3.17))
    ice_permittivity_maetzler06(10e9, FREEZING_POINT) + delta_epsimag
    impure_ice_permittivity_maetzler06(10e9, FREEZING_POINT, 0.013 * PSU)


# Test saline_ice permittivity for mixtures
def test_saline_permittivity_with_mixtures():
    eps_spheres = saline_ice_permittivity_pvs_mixing(18e9, 265, 0)
    eps_needles = saline_ice_permittivity_pvs_mixing(18e9, 265, 0, brine_inclusion_shape="random_needles")

    eps_mix = saline_ice_permittivity_pvs_mixing(
        18e9, 265, 0, brine_inclusion_shape={"spheres": 0.3, "random_needles": 0.7}
    )
    assert abs(eps_mix - (0.3 * eps_spheres + 0.7 * eps_needles)) < 1e-7

    eps_mix = saline_ice_permittivity_pvs_mixing(
        18e9,
        265,
        0,
        brine_inclusion_shape=("spheres", "random_needles"),
        brine_mixing_ratio=0.3,
    )
    assert abs(eps_mix - (0.3 * eps_spheres + 0.7 * eps_needles)) < 1e-7
