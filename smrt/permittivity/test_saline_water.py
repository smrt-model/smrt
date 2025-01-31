import pytest

from smrt.permittivity.saline_water import (
    seawwater_permittivity_boutin21_2function,
    seawwater_permittivity_boutin21_3function,
)
from smrt.core.globalconstants import FREEZING_POINT
from smrt import PSU, GHz


def test_permittivity_boutin21_2function():
    gsw = pytest.importorskip("gsw")

    epsi = seawwater_permittivity_boutin21_2function(1.4 * GHz, 5 + FREEZING_POINT, 33 * PSU)
    assert abs(epsi.real - 76.4080) < 1e-4
    assert abs(epsi.imag - (-50.0570)) < 1e-4


def test_permittivity_boutin21_3function():
    gsw = pytest.importorskip("gsw")

    epsi = seawwater_permittivity_boutin21_3function(1.4 * GHz, 5 + FREEZING_POINT, 33 * PSU)
    assert abs(epsi.real - 76.4710) < 1e-4
    assert abs(epsi.imag - (-50.0611)) < 1e-4
