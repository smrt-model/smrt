import pytest

from smrt import PSU, GHz
from smrt.core.globalconstants import FREEZING_POINT
from smrt.permittivity.saline_water import (
    seawwater_permittivity_boutin23_2function,
    seawwater_permittivity_boutin23_3function,
)


def test_permittivity_boutin23_2function():
    pytest.importorskip("gsw")

    epsi = seawwater_permittivity_boutin23_2function(1.4 * GHz, 5 + FREEZING_POINT, 33 * PSU)
    assert abs(epsi.real - 76.4080) < 1e-4
    assert abs(epsi.imag - (-50.0570)) < 1e-4


def test_permittivity_boutin23_3function():
    pytest.importorskip("gsw")

    epsi = seawwater_permittivity_boutin23_3function(1.4 * GHz, 5 + FREEZING_POINT, 33 * PSU)
    assert abs(epsi.real - 76.4710) < 1e-4
    assert abs(epsi.imag - (-50.0611)) < 1e-4
