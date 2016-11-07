
import numpy as np

from smrt import make_snowpack
from smrt.core.sensor import passive
from smrt.core.model import Model

from smrt.interface.transparent import Transparent
from smrt.emmodel.nonescattering import NoneScattering
from smrt.rtsolver.dort import DORT


def test_noabsoprtion():

    temp = 250
    sp = make_snowpack([100], None, density=[300], temperature=[temp], interface=[Transparent])

    sensor = passive(37e9, theta=[30, 40])

    m = Model(NoneScattering, DORT)
    res = m.run(sensor, sp)

    np.testing.assert_allclose(res.data, temp)
