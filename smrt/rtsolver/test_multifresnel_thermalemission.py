from numpy.testing import assert_allclose

import pytest

from smrt import make_snowpack

from smrt.core.sensor import passive
from smrt.core.model import Model
from smrt.emmodel.nonscattering import NonScattering
from smrt.rtsolver.multifresnel_thermalemission import MultiFresnelThermalEmission
from smrt.rtsolver.dort import DORT


@pytest.fixture
def snowpack():
    return make_snowpack(
        thickness=[10, 10, 10, 20, 30, 3000],
        microstructure_model="homogeneous",
        density=[100, 200, 300, 200, 400, 917],
        temperature=[273, 260, 250, 240, 230, 250],
    )


def test_basic_mfte(snowpack):
    """test MFTE against some values calculated earlier."""
    theta = [30, 40]
    sensor = passive(1.4e9, theta)

    m = Model(NonScattering, MultiFresnelThermalEmission)
    res = m.run(sensor, snowpack)

    print(res.data.coords)
    assert_allclose(res.TbV(), [244.445812, 245.941421])
    assert_allclose(res.TbH(), [240.111885, 237.916166])


def test_mfte_vs_dort(snowpack):
    """test MFTE against DORT."""
    theta = [30, 40]
    sensor = passive(1.4e9, theta)

    m = Model(NonScattering, MultiFresnelThermalEmission)
    res = m.run(sensor, snowpack)

    m2 = Model(NonScattering, DORT)
    res2 = m2.run(sensor, snowpack)

    print(res.data.coords)
    assert_allclose(res.TbV(), res2.TbV(), atol=0.03)
    assert_allclose(res.TbH(), res2.TbH(), atol=0.03)
