import numpy as np

from smrt import make_snowpack
from smrt.core.model import Model
from smrt.core.sensor import active
from smrt.emmodel.nonscattering import NonScattering
from smrt.interface.transparent import Transparent
from smrt.rtsolver.iterative_second_order import IterativeSecondOrder


def setup_snowpack():
    temp = 250
    return make_snowpack([100], "homogeneous", density=[300], temperature=[temp], interface=[Transparent])


def setup_snowpack_with_DH():
    return make_snowpack(
        [0.5, 1000], "homogeneous", density=[300, 250], temperature=2 * [250], interface=2 * [Transparent]
    )


def setup_2layer_snowpack():
    return make_snowpack(
        [0.5, 1000], "homogeneous", density=[250, 300], temperature=2 * [250], interface=2 * [Transparent]
    )


def setup_inf_snowpack():
    temp = 250
    return make_snowpack(
        [10000000], "exponential", corr_length=1e-4, density=[300], temperature=[temp], interface=[Transparent]
    )


def test_returned_theta():
    sp = setup_snowpack()

    theta = [30, 40]
    sensor = active(17.25e9, theta)

    m = Model(NonScattering, IterativeSecondOrder)
    res = m.run(sensor, sp)

    res_theta = res.coords["theta_inc"]
    np.testing.assert_allclose(res_theta, theta)


def test_selectby_theta():
    sp = setup_snowpack()

    theta = [30, 40, 50, 60]
    sensor = active(17.25e9, theta)

    m = Model(NonScattering, IterativeSecondOrder)
    res = m.run(sensor, sp)

    print(res.data.coords)
    res.sigmaVV_dB(theta=50)


def test_depth_hoar_stream_numbers():
    # Will throw error if doesn't run
    sp = setup_snowpack_with_DH()
    sensor = active(13e9, 45)
    m = Model(NonScattering, IterativeSecondOrder)
    m.run(sensor, sp).sigmaVV()


def test_2layer_pack():
    # Will throw error if doesn't run
    sp = setup_2layer_snowpack()
    sensor = active(13e9, 45)
    m = Model(NonScattering, IterativeSecondOrder)
    m.run(sensor, sp).sigmaVV()


def test_normal_call():
    sp = setup_inf_snowpack()

    sensor = active(17.25e9, 30)

    m = Model("iba", "iterative_second_order")
    m.run(sensor, sp)


def test_return_contributions():
    sp = setup_inf_snowpack()

    sensor = active(17.25e9, 30)

    m = Model("iba", "iterative_second_order", rtsolver_options={"return_contributions": True})
    res = m.run(sensor, sp)
    np.testing.assert_allclose(len(res.sigmaVV().contribution), 8)
