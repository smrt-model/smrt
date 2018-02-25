
import numpy as np

from smrt import make_snowpack
from smrt.core.sensor import passive
from smrt.core.model import Model

from smrt.interface.transparent import Transparent
from smrt.emmodel.nonescattering import NoneScattering
from smrt.rtsolver.dort import DORT


def setup_snowpack():
    temp = 250
    return make_snowpack([100], None, density=[300], temperature=[temp], interface=[Transparent])


def test_noabsoprtion():

    sp = setup_snowpack()

    sensor = passive(37e9, theta=[30, 40])
    m = Model(NoneScattering, DORT)
    res = m.run(sensor, sp)

    np.testing.assert_allclose(res.data, sp.layers[0].temperature)


def test_returned_theta():

    sp = setup_snowpack()

    theta = [30, 40]
    sensor = passive(37e9, theta)

    m = Model(NoneScattering, DORT)
    res = m.run(sensor, sp)

    res_theta = res.coords['theta']
    print(res_theta)
    np.testing.assert_allclose(res_theta, theta)


def test_selectby_theta():

    sp = setup_snowpack()

    theta = [30, 40]
    sensor = passive(37e9, theta)

    m = Model(NoneScattering, DORT)
    res = m.run(sensor, sp)

    print(res.data.coords)
    res.TbV(theta=30)
