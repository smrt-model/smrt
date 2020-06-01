

from smrt.core import globalconstants

# Perhaps these are not strictly necessary, but they will provide an extra level of security
# to warn if these constants have been changed.


def test_density_of_ice():
    assert globalconstants.DENSITY_OF_ICE == 916.7


def test_freezing_point():
    assert globalconstants.FREEZING_POINT == 273.15


def test_permittivity_of_air():
    assert globalconstants.PERMITTIVITY_OF_AIR == 1


def test_speed_of_light():
    assert globalconstants.C_SPEED == 2.99792458e8
