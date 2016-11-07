# coding: utf-8

from smrt.emmodel.rayleigh import Rayleigh
from smrt.core.error import SMRTError
from smrt.inputs.sensor_list import amsre
from smrt import make_snow_layer
from smrt.emmodel import commontest

# import the microstructure
from smrt.microstructure_model.independent_sphere import IndependentSphere
tolerance_pc = 0.01  # 5% tolerance


def setup_func_sp():
    # ### Make a snow layer
    shs_lay = make_snow_layer(layer_thickness=0.2, microstructure_model=IndependentSphere, density=250, temperature=265, radius=5e-4)
    return shs_lay


def setup_func_rad(radius):
    # ### Make a snow layer
    shs_lay = make_snow_layer(layer_thickness=0.1, microstructure_model=IndependentSphere, density=300, temperature=265, radius=radius)
    return shs_lay


def setup_func_em(testpack=None):
    if testpack is None:
        testpack = setup_func_sp()
    sensor = amsre('37V')
    emmodel = Rayleigh(sensor, testpack)
    return emmodel


def test_energy_conservation():

    em = setup_func_em()
    commontest.test_energy_conservation(em, tolerance_pc)
