# coding: utf-8
import pytest

from smrt.emmodel import commontest
from smrt.emmodel.prescribed_kskaeps import Prescribed_KsKaEps
from smrt.inputs.make_medium import make_generic_layer
from smrt.inputs.sensor_list import amsre

tolerance_pc = 0.01  # 1% tolerance


@pytest.fixture
def setup_func_sp():
    # ### Make a generic layer
    lay = make_generic_layer(layer_thickness=0.2, ks=0.1, ka=0.2, effective_permittivity=1.3)

    return lay


def setup_func_em(setup_func_sp, testpack=None):
    if testpack is None:
        testpack = setup_func_sp
    sensor = amsre("37V")
    emmodel = Prescribed_KsKaEps(sensor, testpack)
    return emmodel


def test_energy_conservation(setup_func_sp):
    em = setup_func_em(setup_func_sp)
    commontest.test_energy_conservation(em, tolerance_pc)
