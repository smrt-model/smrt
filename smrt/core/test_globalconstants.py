from smrt.core import globalconstants
import pytest

# Perhaps these are not strictly necessary, but they will provide an extra level of security
# to warn if these constants have been changed.

@pytest.mark.parametrize("name, constant, expected",
                         [("DENSITY_OF_ICE", globalconstants.DENSITY_OF_ICE, 916.7),        ("FREEZING_POINT", globalconstants.FREEZING_POINT, 273.15),        ("PERMITTIVITY_OF_AIR", globalconstants.PERMITTIVITY_OF_AIR, 1),("C_SPEED", globalconstants.C_SPEED, 2.99792458e8)])
def test_constant_named(name, constant, expected):
    assert constant == expected, f"{name} expected {expected}, got {constant}"