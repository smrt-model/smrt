# coding: utf-8

import numpy as np

# local import
from smrt import make_snowpack, make_model, make_soil, sensor_list
from smrt.atmosphere.simple_isotropic_atmosphere import SimpleIsotropicAtmosphere


def test_isothermal_universe_highscatt():
    _do_test_isothermal_universe(0.8e-3, 10)


def test_isothermal_universe_lowscatt():
    _do_test_isothermal_universe(0.05e-3, 10)


def test_isothermal_universe_shallow():
    _do_test_isothermal_universe(0.8e-3, 0.1)


def test_kirchoff_law_highscatt():
    _do_test_kirchoff_law(0.8e-3, 10)


def test_kirchoff_law_lowscatt():
    _do_test_kirchoff_law(0.05e-3, 10)


def test_kirchoff_law_shallow():
    _do_test_kirchoff_law(0.8e-3, 0.1)


def _do_test_isothermal_universe(pc, thickness_1):

    T = 265

    substrate = make_soil('soil_wegmuller', permittivity_model=complex(10, 1), roughness_rms=0.001, temperature=T)

    atmosphere = SimpleIsotropicAtmosphere(tbdown=T, tbup=0, trans=1)

    snowpack = make_snowpack([0.3, thickness_1], "exponential",
                             density=[200, 300], temperature=T, corr_length=pc,
                             ice_permittivity_model=complex(1.7, 0.00001),
                             substrate=substrate, atmosphere=atmosphere)

    # create the sensor
    theta = range(10, 80, 5)
    radiometer = sensor_list.passive(37e9, theta)

    # create the EM Model
    m = make_model("iba", "dort")

    # run the model
    sresult = m.run(radiometer, snowpack)

    np.testing.assert_allclose(sresult.TbV(), T, atol=0.01)
    np.testing.assert_allclose(sresult.TbH(), T, atol=0.01)


def _do_test_kirchoff_law(pc, thickness_1):

    T = 265.

    substrate = make_soil('soil_wegmuller', permittivity_model=complex(10, 1), roughness_rms=0.001, temperature=T)

    atmosphere1K = SimpleIsotropicAtmosphere(tbdown=1, tbup=0, trans=1)

    snowpack = make_snowpack([0.3, thickness_1], "exponential",
                             density=[200, 300], temperature=T, corr_length=pc,
                             ice_permittivity_model=complex(1.7, 0.00001),
                             substrate=substrate)

    # create the sensor
    theta = range(10, 80, 5)
    radiometer = sensor_list.passive(37e9, theta)

    # create the EM Model
    m = make_model("iba", "dort")

    # run the model
    sresult_0 = m.run(radiometer, snowpack)
    snowpack.atmosphere = atmosphere1K
    sresult_1 = m.run(radiometer, snowpack)

    # V-pol
    emissivity_V = (sresult_0.TbV() + sresult_1.TbV()) / 2 / T
    reflectivity_V = (sresult_1.TbV() - sresult_0.TbV())

    print(emissivity_V, 1 - reflectivity_V)
    np.testing.assert_allclose(emissivity_V, 1 - reflectivity_V, atol=0.002)

    # H-pol
    emissivity_H = (sresult_0.TbH() + sresult_1.TbH()) / 2 / T
    reflectivity_H = (sresult_1.TbH() - sresult_0.TbH())

    print(emissivity_H, 1 - reflectivity_H)
    np.testing.assert_allclose(emissivity_H, 1 - reflectivity_H, atol=0.002)
