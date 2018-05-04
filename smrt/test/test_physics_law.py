# coding: utf-8

import numpy as np
from nose.tools import ok_

# local import
from smrt import make_snowpack, make_model, make_soil, sensor_list
from smrt.atmosphere.simple_isotropic_atmosphere import SimpleIsotropicAtmosphere


def test_isothermal_universe():

    # prepare snowpack
    pc=0.8e-3

    T = 265

    substrate = make_soil('soil_wegmuller',permittivity_model=complex(10,1), roughness_rms=0.001, temperature=T)

    snowpack = make_snowpack([0.5, 10], "exponential",
                            density=[200, 300], temperature=T, corr_length=pc, substrate=substrate)

    atmosphere = SimpleIsotropicAtmosphere(tbdown=T, tbup=0, trans=1)

    # create the sensor
    theta = range(10, 80, 5)
    radiometer = sensor_list.passive(37e9, theta)

    # create the EM Model
    m = make_model("iba", "dort")

    # run the model
    sresult = m.run(radiometer, snowpack, atmosphere)

    np.testing.assert_allclose(sresult.TbV(), T, atol=0.01)
    np.testing.assert_allclose(sresult.TbH(), T, atol=0.01)
