# coding: utf-8

from smrt.emmodel.sft_rayleigh import SFT_Rayleigh
from smrt.inputs.sensor_list import amsre
from smrt import make_snow_layer
from smrt.emmodel import commontest

# import the microstructure
from smrt.microstructure_model.exponential import Exponential
tolerance_pc = 0.01  # 1% tolerance


def setup_func_sp():
    # ### Make a snow layer
    lay = make_snow_layer(layer_thickness=0.2, microstructure_model=Exponential, density=250, temperature=265, corr_length=5e-4)
    return lay


def setup_func_em(testpack=None):
    if testpack is None:
        testpack = setup_func_sp()
    sensor = amsre('37V')
    emmodel = SFT_Rayleigh(sensor, testpack)
    return emmodel


def test_energy_conservation():

    em = setup_func_em()
    commontest.test_energy_conservation(em, tolerance_pc)
