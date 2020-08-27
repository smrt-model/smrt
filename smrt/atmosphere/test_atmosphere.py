

import numpy as np

from smrt import make_snowpack, make_model, sensor_list
from smrt.atmosphere.simple_isotropic_atmosphere import SimpleIsotropicAtmosphere, make_atmosphere


def test_simple_isotropic_atmosphere():

    # prepare inputs
    density = [300, 300]
    temperature = [265, 265]
    thickness = [0.4, 10]
    radius = [200e-6, 400e-6]
    stickiness = [0.2, 0.2]

    rads = sensor_list.amsre('36V')

    atmos = SimpleIsotropicAtmosphere(30., 6., 0.90)

    snowpack = make_snowpack(thickness, "sticky_hard_spheres", density=density,
                             temperature=temperature, radius=radius, stickiness=stickiness)

    # create the EM Model - Equivalent DMRTML
    iba = make_model("iba", "dort")

    res1 = iba.run(rads, snowpack)

    snowpack.atmosphere = atmos
    res2 = iba.run(rads, snowpack)

    print('TB 1: ', res1.TbV(), 'TB2: ', res2.TbV())

    # absorption with effective permittivity
    assert abs(res1.TbV() - 227.61318467710458) < 1e-2
    assert abs(res2.TbV() - 214.66092232541834) < 1e-2


def test_frequency_dependent_atmosphere():

    mu = np.cos(np.arange(0, 90))
    atmos = make_atmosphere(tbdown={10e9: 15, 21e9: 23}, tbup={10e9: 5, 21e9: 6}, trans={10e9: 1, 21e9: 0.95})

    assert np.all(atmos.tbup(frequency=10e9, costheta=mu, npol=2) == 5)
    assert np.all(atmos.tbdown(frequency=21e9, costheta=mu, npol=2) == 23)
    assert np.all(atmos.trans(frequency=21e9, costheta=mu, npol=2) == 0.95)
