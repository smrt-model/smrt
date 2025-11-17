import warnings

import numpy as np
import pytest

from smrt import make_model, make_snowpack, make_soil, sensor_list
from smrt.core.error import SMRTWarning
from smrt.core.fresnel import snell_angle
from smrt.core.model import Model
from smrt.core.sensor import active
from smrt.emmodel.iba import IBA
from smrt.emmodel.nonscattering import NonScattering
from smrt.interface.transparent import Transparent
from smrt.rtsolver.iterative_first_order import IterativeFirstOrder


@pytest.fixture
def setup_snowpack():
    temp = 250
    return make_snowpack([100], "homogeneous", density=[300], temperature=[temp], interface=[Transparent])


@pytest.fixture
def setup_snowpack_with_DH():
    return make_snowpack(
        [0.5, 1000], "homogeneous", density=[300, 250], temperature=2 * [250], interface=2 * [Transparent]
    )


@pytest.fixture
def setup_2layer_snowpack():
    return make_snowpack(
        [0.5, 1000], "homogeneous", density=[250, 300], temperature=2 * [250], interface=2 * [Transparent]
    )


@pytest.fixture
def setup_inf_snowpack():
    temp = 250
    return make_snowpack(
        [10000000], "exponential", corr_length=1e-4, density=[300], temperature=[temp], interface=[Transparent]
    )


def test_returned_theta(setup_snowpack):
    sp = setup_snowpack

    theta = [30, 40]
    sensor = active(17.25e9, theta)

    m = Model(NonScattering, IterativeFirstOrder)
    res = m.run(sensor, sp)

    res_theta = res.coords["theta_inc"]
    print(res_theta)
    np.testing.assert_allclose(res_theta, theta)
    print(res.data.coords)
    res.sigmaVV_dB(theta=30)


# Parametrization of the two following tests failed
def test_depth_hoar_stream_numbers(setup_snowpack_with_DH):
    # Will throw error if doesn't run
    sp = setup_snowpack_with_DH
    sensor = active(13e9, 45)
    m = Model(NonScattering, IterativeFirstOrder)
    m.run(sensor, sp).sigmaVV()


def test_2layer_pack(setup_2layer_snowpack):
    # Will throw error if doesn't run
    sp = setup_2layer_snowpack
    sensor = active(13e9, 45)
    m = Model(NonScattering, IterativeFirstOrder)
    m.run(sensor, sp).sigmaVV()


def test_shallow_snowpack():
    warnings.filterwarnings("error", message=".*optically shallow.*", module=".*iterative_first_order")

    with pytest.raises(SMRTWarning):
        sp = make_snowpack(
            [0.15, 0.15], "homogeneous", density=[300, 250], temperature=2 * [250], interface=2 * [Transparent]
        )
        sensor = active(17e9, 45)
        m = Model(NonScattering, IterativeFirstOrder)
        m.run(sensor, sp, parallel_computation=False).sigmaVV()


def test_infinite_pack(setup_inf_snowpack):
    # from Ghi thesis p43, setup an infinite snowpack with transparent interface
    # also from ulaby et al 2014 (11.75) d -> inf so gamma2 --> 0
    # in this specific case
    # sigma = (1 - 0)/(2 * ke)  * (phase function * (T @ I_i * dense_factor))
    # interface transparent so T is 1
    # I_i = 1 for VV and HH
    sp = setup_inf_snowpack
    theta = [30]
    sensor = active(17.25e9, theta)

    m = Model("iba", IterativeFirstOrder)
    res = m.run(sensor, sp)
    ke = res.optical_depth() / sp.layer_thicknesses

    mu = np.cos(sensor.theta)
    emmodels = IBA(sensor, sp.layers[0])
    # direct backscatter (phase function * 1 * dense snow factor)
    specific_intensity_conversion = (1 / emmodels.effective_permittivity().real) * (
        mu / snell_angle(1, emmodels.effective_permittivity().real, mu)
    )
    phase = emmodels.phase(-mu, mu, np.pi, 2).values.squeeze() * specific_intensity_conversion

    # should be equal to the phase function * 1
    scattering_test = res.sigmaVV() * 2 * ke / mu
    np.testing.assert_allclose(phase[0, 0], scattering_test)


def test_normal_call(setup_snowpack):
    sp = setup_snowpack

    theta = [30, 40]
    sensor = active(17.25e9, theta)

    m = Model(NonScattering, "iterative_first_order")
    m.run(sensor, sp)


def test_return_contributions(setup_snowpack):
    sp = setup_snowpack

    sensor = active(17.25e9, 30)

    m = Model(NonScattering, "iterative_first_order", rtsolver_options={"return_contributions": True})
    res = m.run(sensor, sp)
    np.testing.assert_allclose(len(res.sigmaVV().contribution), 5)


def test_all_substrate():
    thick = [2]  # m
    temperature = [270]  # kelvin
    density = np.array([250])  # kg/m^3
    corr_length = [8e-5]  # 80 microns

    substrates = [
        make_soil("flat", complex(10, 0.005), temperature=265),
        make_soil("geometrical_optics_backscatter", complex(2, 0.005), mean_square_slope=0.02, temperature=265),
        make_soil("geometrical_optics", complex(5, 0.005), mean_square_slope=0.2, temperature=265),
        make_soil("iem_fung92", complex(5, 0.005), roughness_rms=0.0001, corr_length=0.005, temperature=265),
        make_soil("iem_fung92_brogioni10", complex(5, 0.005), roughness_rms=0.001, corr_length=0.005, temperature=265),
    ]

    for substrate in substrates:
        # create the snowpack
        snowpack = make_snowpack(
            thickness=thick,
            microstructure_model="exponential",
            density=density,
            temperature=temperature,
            corr_length=corr_length,
            substrate=substrate,
        )

        sensor = sensor_list.active(17.25e9, 30)
        # test with DORT and compare with final sigma
        model = make_model("iba", "iterative_first_order", rtsolver_options={"error_handling": "nan"})
        model.run(sensor, snowpack)
