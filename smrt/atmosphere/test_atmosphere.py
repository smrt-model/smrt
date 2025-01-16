import pytest
import numpy as np

from smrt import make_snowpack, make_model, sensor_list
from smrt.atmosphere.simple_isotropic_atmosphere import SimpleIsotropicAtmosphere
from smrt.atmosphere.simple_atmosphere import SimpleAtmosphere


@pytest.fixture()
def default_snowpack():
    # prepare inputs
    density = [300, 300]
    temperature = [265, 265]
    thickness = [0.4, 10]
    radius = [200e-6, 400e-6]
    stickiness = [0.2, 0.2]

    snowpack = make_snowpack(
        thickness, "sticky_hard_spheres", density=density, temperature=temperature, radius=radius, stickiness=stickiness
    )
    return snowpack


def test_simple_isotropic_atmosphere(default_snowpack):

    atmos = SimpleIsotropicAtmosphere(tb_down=30.0, tb_up=6.0, transmittance=0.90)

    snowpack = atmos + default_snowpack

    # create the EM Model - Equivalent DMRTML
    rads = sensor_list.amsre("36V")
    m = make_model("iba", "dort")

    res1 = m.run(rads, default_snowpack)
    res2 = m.run(rads, snowpack)

    print("TB 1: ", res1.TbV(), "TB2: ", res2.TbV())

    # absorption with effective permittivity
    assert abs(res1.TbV() - 227.61318467710458) < 1e-2
    assert abs(res2.TbV() - 214.66092232541834) < 1e-2


def test_simple_atmosphere(default_snowpack):
    rads = sensor_list.amsre("36V")

    atmos = SimpleAtmosphere(
        theta=[0, 45, 90], tb_down=[23, 28, 33], tb_up=[20, 25, 30.0], transmittance=[0.85, 0.85, 0.90]
    )

    snowpack = atmos + default_snowpack

    # create the EM Model - Equivalent DMRTML
    iba = make_model("iba", "dort")

    res1 = iba.run(rads, default_snowpack)
    res2 = iba.run(rads, snowpack)

    print("TB 1: ", res1.TbV(), "TB2: ", res2.TbV())

    # absorption with effective permittivity
    assert abs(res1.TbV() - 227.61318467710458) < 1e-2
    assert abs(res2.TbV() - 224.9027432887187) < 1e-2


def test_frequency_dependent_atmosphere():
    mu = np.cos(np.arange(0, 90))
    atmos = SimpleIsotropicAtmosphere(
        tb_down={10e9: 15, 21e9: 23}, tb_up={10e9: 5, 21e9: 6}, transmittance={10e9: 1, 21e9: 0.95}
    )

    assert np.all(atmos.run(frequency=10e9, costheta=mu, npol=2).tb_up == 5)
    assert np.all(atmos.run(frequency=21e9, costheta=mu, npol=2).tb_down == 23)
    assert np.all(atmos.run(frequency=21e9, costheta=mu, npol=2).transmittance == 0.95)


def test_dict_param_atmosphere():
    # test if one atmo param can be dict or other not specify (defaut)

    mu = np.cos(np.arange(0, 90))
    atmos = SimpleIsotropicAtmosphere(tb_down={10e9: 15, 21e9: 23})

    assert np.all(atmos.run(frequency=21e9, costheta=mu, npol=2).tb_down == 23)
    assert np.all(atmos.run(frequency=10e9, costheta=mu, npol=2).tb_down == 15)
    assert np.all(atmos.run(frequency=21e9, costheta=mu, npol=2).tb_up == 0)
    assert np.all(atmos.run(frequency=10e9, costheta=mu, npol=2).tb_up == 0)
    assert np.all(atmos.run(frequency=21e9, costheta=mu, npol=2).transmittance == 1)
    assert np.all(atmos.run(frequency=10e9, costheta=mu, npol=2).transmittance == 1)
