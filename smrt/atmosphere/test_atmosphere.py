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


@pytest.fixture()
def atmosphere1():
    return SimpleIsotropicAtmosphere(tb_down=20.0, tb_up=6.0, transmittance=0.90)


@pytest.fixture()
def atmosphere2():
    return SimpleIsotropicAtmosphere(tb_down=10.0, tb_up=4.0, transmittance=0.80)


@pytest.fixture()
def atmosphere3():
    return SimpleIsotropicAtmosphere(tb_down=5.0, tb_up=2.0, transmittance=0.70)


def test_tb_access(atmosphere1):
    res = atmosphere1.run(frequency=10e9, costheta=np.array([1.0]), npol=1, rayleigh_jeans_approximation=True)

    np.testing.assert_equal(res.tb_down, res.intensity_down)
    np.testing.assert_equal(res.tb_up, res.intensity_up)


@pytest.mark.parametrize("rayleigh_jeans_approximation", [False, True])
def test_simple_isotropic_atmosphere(default_snowpack, rayleigh_jeans_approximation):
    atmos = SimpleIsotropicAtmosphere(tb_down=30.0, tb_up=6.0, transmittance=0.90)

    snowpack = atmos + default_snowpack

    # create the EM Model - Equivalent DMRTML
    rads = sensor_list.amsre("36V")
    m = make_model("iba", "dort", rtsolver_options=dict(rayleigh_jeans_approximation=rayleigh_jeans_approximation))

    res1 = m.run(rads, default_snowpack)
    res2 = m.run(rads, snowpack)

    print("TB 1: ", res1.TbV(), "TB2: ", res2.TbV())

    # absorption with effective permittivity
    if rayleigh_jeans_approximation:
        np.testing.assert_allclose(res1.TbV(), 227.61002775786866, atol=1e-2)
        np.testing.assert_allclose(res2.TbV(), 214.65840930416707, atol=1e-2)
    else:
        np.testing.assert_allclose(res1.TbV(), 227.73331276273777, atol=1e-2)
        np.testing.assert_allclose(res2.TbV(), 213.9133292330192, atol=1e-2)


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
    res2 = iba.run(
        rads,
        snowpack,
    )

    print("TB 1: ", res1.TbV(), "TB2: ", res2.TbV())

    # absorption with effective permittivity
    np.testing.assert_allclose(res1.TbV(), 227.73331276273777, atol=1e-2)
    np.testing.assert_allclose(res2.TbV(), 224.16055686943304, atol=1e-2)


def test_frequency_dependent_atmosphere():
    mu = np.cos(np.arange(0, 90))
    atmos = SimpleIsotropicAtmosphere(
        tb_down={10e9: 15, 21e9: 23},
        tb_up={10e9: 5, 21e9: 6},
        transmittance={10e9: 1, 21e9: 0.95},
    )

    kwargs = dict(costheta=mu, npol=2, rayleigh_jeans_approximation=True)

    np.testing.assert_equal(atmos.run(frequency=10e9, **kwargs).tb_up, 5)
    np.testing.assert_equal(atmos.run(frequency=21e9, **kwargs).tb_down, 23)
    np.testing.assert_equal(atmos.run(frequency=21e9, **kwargs).transmittance, 0.95)


def test_dict_param_atmosphere():
    # test if one atmo param can be dict or other not specify (defaut)

    mu = np.cos(np.arange(0, 90))
    atmos = SimpleIsotropicAtmosphere(tb_down={10e9: 15, 21e9: 23})

    kwargs = dict(costheta=mu, npol=2, rayleigh_jeans_approximation=True)
    np.testing.assert_equal(atmos.run(frequency=21e9, **kwargs).tb_down, 23)
    np.testing.assert_equal(atmos.run(frequency=10e9, **kwargs).tb_down, 15)
    np.testing.assert_equal(atmos.run(frequency=21e9, **kwargs).tb_up, 0)
    np.testing.assert_equal(atmos.run(frequency=10e9, **kwargs).tb_up, 0)
    np.testing.assert_equal(atmos.run(frequency=21e9, **kwargs).transmittance, 1)
    np.testing.assert_equal(atmos.run(frequency=10e9, **kwargs).transmittance, 1)


def test_adding_atmospheres(atmosphere1, atmosphere2):
    stacked_atmos = atmosphere1 + atmosphere2

    assert isinstance(stacked_atmos, AtmosphereStack)

    res = stacked_atmos.run(frequency=10e9, costheta=np.array([1.0]), npol=1, rayleigh_jeans_approximation=True)

    np.testing.assert_allclose(res.intensity_down, (20.0 * 0.80 + 10.0), atol=1e-6)
    np.testing.assert_allclose(res.intensity_up, (6.0 + 0.90 * 4.0), atol=1e-6)
    np.testing.assert_allclose(res.transmittance, (0.90 * 0.80), atol=1e-6)


def test_inplace_adding_atmospheres(atmosphere1, atmosphere2, atmosphere3):
    stacked_atmos = atmosphere1 + atmosphere2

    stacked_atmos += atmosphere3

    res = stacked_atmos.run(frequency=10e9, costheta=np.array([1.0]), npol=1, rayleigh_jeans_approximation=True)

    np.testing.assert_allclose(res.intensity_down, ((20.0 * 0.80 + 10.0) * 0.70 + 5.0), atol=1e-6)
    np.testing.assert_allclose(res.intensity_up, (6.0 + 0.90 * 4.0 + 0.80 * 0.90 * 2.0), atol=1e-6)
    np.testing.assert_allclose(res.transmittance, (0.90 * 0.80 * 0.70), atol=1e-6)
