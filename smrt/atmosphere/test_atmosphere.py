

import numpy as np
from nose.tools import ok_

from smrt import make_snowpack, make_model, sensor, make_soil
from smrt import sensor_list
from smrt.atmosphere.simple_isotropic_atmosphere import SimpleIsotropicAtmosphere


def test_simple_isotropic_atmosphere():

    # prepare inputs
    density = [300, 300]
    temperature = [265, 265]
    thickness = [0.4, 10]
    radius = [200e-6, 400e-6]
    #stickiness = [1000, 1000]
    stickiness = [0.2, 0.2]

    rads = sensor_list.amsre('36V')

    atmos = SimpleIsotropicAtmosphere(30., 6., 0.90)

    snowpack = make_snowpack(thickness, "sticky_hard_spheres",
                            density=density, temperature=temperature, radius=radius, stickiness=stickiness)

    # create the EM Model - Equivalent DMRTML
    iba = make_model("iba", "dort")

    res1 = iba.run(rads, snowpack)
    res2 = iba.run(rads, snowpack, atmosphere=atmos)

    print('TB 1: ', res1.TbV(), 'TB2: ', res2.TbV())

    #absorption with effective permittivity
    ok_(abs(res1.TbV() - 227.61318467710458) < 1e-2)
    ok_(abs(res2.TbV() - 214.66092232541834) < 1e-2)

    #original absorption (Maetzler 1998)
    #ok_(abs(res1.TbV() - 223.925496277253) < 1e-2)
    #ok_(abs(res2.TbV() - 211.7175839349701) < 1e-2)
