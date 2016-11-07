from nose.tools import eq_

from smrt.core import globalconstants

# Perhaps these are not strictly necessary, but they will provide an extra level of security
# to warn if these constants have been changed.


def test_density_of_ice():
    eq_(globalconstants.DENSITY_OF_ICE, 917)


def test_density_of_ice():
    eq_(globalconstants.FREEZING_POINT, 273.15)


def test_permittivity_of_air():
    eq_(globalconstants.PERMITTIVITY_OF_AIR, 1)


def test_speed_of_light():
    eq_(globalconstants.C_SPEED, 2.99792458e8)