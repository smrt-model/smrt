
from nose.tools import eq_
import numpy as np

from smrt.permittivity.ice import ice_permittivity_maetzler06
from smrt.permittivity.saline_ice import impure_ice_permittivity_maetzler06
from smrt.core.globalconstants import FREEZING_POINT


# Test impure ice with zero salinity same as pure ice permittivity
def test_impure_permittivity_same_as_pure_for_zero_salinty():
    pure = ice_permittivity_maetzler06(18e9, 265)
    impure = impure_ice_permittivity_maetzler06(18e9, 265, 0)
    eq_(pure, impure)


# Test of impure ice at freezing point temperature, 0.013PSU salinity, 10GHz
def test_impure_ice_freezing_point_0p013psu_10GHz():
    delta_epsimag = 1. / (1866 * np.exp(-3.17))
    test_eps = ice_permittivity_maetzler06(10e9, FREEZING_POINT) + delta_epsimag
    impure = impure_ice_permittivity_maetzler06(10e9, FREEZING_POINT, 0.013)
