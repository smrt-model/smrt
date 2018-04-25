
import numpy as np

from smrt import make_snowpack
from smrt.core.sensor import passive, active
from smrt.core.model import Model

from smrt.interface.transparent import Transparent
from smrt.emmodel.nonscattering import NonScattering
from smrt.rtsolver.dort import DORT


def setup_snowpack():
    temp = 250
    return make_snowpack([100], "homogeneous", density=[300], temperature=[temp], interface=[Transparent])


def setup_snowpack_with_DH():
    return make_snowpack(2*[0.5], "homogeneous", density=[300, 250], temperature=2*[250], interface=2*[Transparent])


def setup_2layer_snowpack():
    return make_snowpack(2*[0.5], "homogeneous", density=[250, 300], temperature=2*[250], interface=2*[Transparent])


def test_noabsoprtion():

    sp = setup_snowpack()

    sensor = passive(37e9, theta=[30, 40])
    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)

    np.testing.assert_allclose(res.data, sp.layers[0].temperature)


def test_returned_theta():

    sp = setup_snowpack()

    theta = [30, 40]
    sensor = passive(37e9, theta)

    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)

    res_theta = res.coords['theta']
    print(res_theta)
    np.testing.assert_allclose(res_theta, theta)


def test_selectby_theta():

    sp = setup_snowpack()

    theta = [30, 40]
    sensor = passive(37e9, theta)

    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)

    print(res.data.coords)
    res.TbV(theta=30)


def test_depth_hoar_stream_numbers():
    # Will throw error if doesn't run
    sp = setup_snowpack_with_DH()
    sensor = active(13e9, 45)
    m = Model(NonScattering, DORT)
    m.run(sensor, sp).sigmaVV()


def test_2layer_pack():
    # Will throw error if doesn't run
    sp = setup_2layer_snowpack()
    sensor = active(13e9, 45)
    m = Model(NonScattering, DORT)
    m.run(sensor, sp).sigmaVV()
