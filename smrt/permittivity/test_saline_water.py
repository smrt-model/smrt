import pytest

from smrt import PSU, GHz
from smrt.core.globalconstants import FREEZING_POINT
from smrt.permittivity.saline_water import (
    seawwater_permittivity_boutin23_2function,
    seawwater_permittivity_boutin23_3function,
)


@pytest.mark.parametrize(
    "permittivity_function, exp_epsi_r, exp_epsi_i",
    [
        (seawwater_permittivity_boutin23_2function, 76.4080, -50.0570),
        (seawwater_permittivity_boutin23_3function, 76.4710, -50.0611),
    ],
)
def test_permittivity_boutin23_2function(permittivity_function, exp_epsi_r, exp_epsi_i):
    pytest.importorskip("gsw")

    epsi = permittivity_function(1.4 * GHz, 5 + FREEZING_POINT, 33 * PSU)
    assert abs(epsi.real - exp_epsi_r) < 1e-4
    assert abs(epsi.imag - exp_epsi_i) < 1e-4
