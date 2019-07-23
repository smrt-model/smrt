
import numpy as np

from smrt.permittivity.ice import ice_permittivity_maetzler06
from smrt.permittivity.saline_ice import impure_ice_permittivity_maetzler06, saline_ice_permittivity_pvs_mixing
from smrt.core.globalconstants import FREEZING_POINT
from smrt import PSU


# Test impure ice with zero salinity same as pure ice permittivity
def test_impure_permittivity_same_as_pure_for_zero_salinty():
    pure = ice_permittivity_maetzler06(18e9, 265)
    impure = impure_ice_permittivity_maetzler06(18e9, 265, 0)
    assert abs(pure.real - impure.real)  < 1e-4
    assert abs(pure.imag- impure.imag) < 1e-7


# Test of impure ice at freezing point temperature, 0.013PSU salinity, 10GHz
def test_impure_ice_freezing_point_0p013psu_10GHz():
    delta_epsimag = 1. / (1866 * np.exp(-3.17))
    test_eps = ice_permittivity_maetzler06(10e9, FREEZING_POINT) + delta_epsimag
    impure = impure_ice_permittivity_maetzler06(10e9, FREEZING_POINT, 0.013*PSU)
    

# Test impure ice with zero salinity same as pure ice permittivity, when using mixing formulas:
def test_saline_permittivity_same_as_pure_for_zero_salinity():
    pure = ice_permittivity_maetzler06(18e9, 265)
    impure = saline_ice_permittivity_pvs_mixing(18e9,265,0)
    assert abs(pure.real - impure.real)  < 1e-4
    assert abs(pure.imag- impure.imag) < 1e-7

# Test saline_ice permittivity for mixtures
def test_saline_permittivity_with_mixtures():
    
    eps_spheres = saline_ice_permittivity_pvs_mixing(18e9,265,0)
    eps_needles = saline_ice_permittivity_pvs_mixing(18e9,265,0, brine_inclusion_shape="random_needles")

    eps_mix = saline_ice_permittivity_pvs_mixing(18e9,265,0, brine_inclusion_shape={"spheres": 0.3, "random_needles": 0.7})
    assert abs(eps_mix - (0.3 * eps_spheres + 0.7 * eps_needles)) < 1e-7

    eps_mix = saline_ice_permittivity_pvs_mixing(18e9,265,0, brine_inclusion_shape=("spheres", "random_needles"), brine_mixing_ratio=0.3)
    assert abs(eps_mix - (0.3 * eps_spheres + 0.7 * eps_needles)) < 1e-7
