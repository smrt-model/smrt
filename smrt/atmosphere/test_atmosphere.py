import numpy as np
import pytest

from smrt import make_model, make_snowpack, sensor_list
from smrt.atmosphere.simple_atmosphere import SimpleAtmosphere
from smrt.atmosphere.simple_isotropic_atmosphere import SimpleIsotropicAtmosphere
from smrt.core.atmosphere import AtmosphereStack


@pytest.fixture()
def default_snowpack():
    # prepare inputs
    density = [300, 300]
    temperature = [265, 265]
    thickness = [0.4, 10]
    radius = [200e-6, 400e-6]
    stickiness = [0.2, 0.2]

    snowpack = make_snowpack(
        thickness,
        "sticky_hard_spheres",
        density=density,
        temperature=temperature,
        radius=radius,
        stickiness=stickiness,
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
    assert abs(res1.TbV() - 227.73331276273777) < 1e-2
    assert abs(res2.TbV() - 213.9133292330192) < 1e-2


def test_simple_atmosphere(default_snowpack):
    rads = sensor_list.amsre("36V")

    atmos = SimpleAtmosphere(
        theta=[0, 45, 90],
        tb_down=[23, 28, 33],
        tb_up=[20, 25, 30.0],
        transmittance=[0.85, 0.85, 0.90],
    )

    snowpack = atmos + default_snowpack

    # create the EM Model - Equivalent DMRTML
    iba = make_model("iba", "dort")

    res1 = iba.run(rads, default_snowpack)
    res2 = iba.run(rads, snowpack)

    print("TB 1: ", res1.TbV(), "TB2: ", res2.TbV())

    # absorption with effective permittivity
    assert abs(res1.TbV() - 227.73331276273777) < 1e-2
    assert abs(res2.TbV() - 224.16055686943304) < 1e-2


def test_frequency_dependent_atmosphere():
    mu = np.cos(np.arange(0, 90))
    atmos = SimpleIsotropicAtmosphere(
        tb_down={10e9: 15, 21e9: 23},
        tb_up={10e9: 5, 21e9: 6},
        transmittance={10e9: 1, 21e9: 0.95},
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


def test_adding_atmospheres():
    atmos1 = SimpleIsotropicAtmosphere(tb_down=20.0, tb_up=6.0, transmittance=0.90)
    atmos2 = SimpleIsotropicAtmosphere(tb_down=10.0, tb_up=4.0, transmittance=0.80)

    stacked_atmos = atmos1 + atmos2

    assert isinstance(stacked_atmos, AtmosphereStack)

    res = stacked_atmos.run(frequency=10e9, costheta=np.array([1.0]), npol=1)

    assert abs(res.tb_down - (20.0 * 0.80 + 10.0)) < 1e-6
    assert abs(res.tb_up - (6.0 + 0.90 * 4.0)) < 1e-6
    assert abs(res.transmittance - (0.90 * 0.80)) < 1e-6


def test_inplace_adding_atmospheres():
    atmos1 = SimpleIsotropicAtmosphere(tb_down=20.0, tb_up=6.0, transmittance=0.90)
    atmos2 = SimpleIsotropicAtmosphere(tb_down=10.0, tb_up=4.0, transmittance=0.80)
    atmos3 = SimpleIsotropicAtmosphere(tb_down=5.0, tb_up=2.0, transmittance=0.70)

    stacked_atmos = atmos1 + atmos2

    stacked_atmos += atmos3

    res = stacked_atmos.run(frequency=10e9, costheta=np.array([1.0]), npol=1)

    assert abs(res.tb_down - ((20.0 * 0.80 + 10.0) * 0.70 + 5.0)) < 1e-6
    assert abs(res.tb_up - (6.0 + 0.90 * 4.0 + 0.80 * 0.90 * 2.0)) < 1e-6
    assert abs(res.transmittance - (0.90 * 0.80 * 0.70)) < 1e-6
