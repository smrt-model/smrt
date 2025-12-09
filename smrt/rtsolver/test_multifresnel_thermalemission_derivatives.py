import pytest
from numpy.testing import assert_allclose

from smrt import make_snowpack
from smrt.core.model import Model
from smrt.core.sensor import passive
from smrt.emmodel.nonscattering import NonScattering
from smrt.rtsolver.multifresnel_thermalemission_derivatives import MultiFresnelThermalEmissionDerivatives


@pytest.fixture
def snowpack():
    return make_snowpack(
        thickness=[10, 10, 10, 20, 30, 3000],
        microstructure_model="homogeneous",
        density=[100, 200, 300, 200, 400, 917],
        temperature=[273, 260, 250, 240, 230, 250],
    )


def test_mfte_derivatives(snowpack):
    """test MFTE against some values calculated earlier."""
    theta = [30, 40]
    sensor = passive(1.4e9, theta)

    m = Model(NonScattering, MultiFresnelThermalEmissionDerivatives)
    res = m.run(sensor, snowpack)

    print(res.data.coords)
    assert_allclose(res.TbV(), [244.445812, 245.941421])
    assert_allclose(res.TbH(), [240.111885, 237.916166])
    assert res.other_data["dTBvsdTi"].values.shape == (6, 2)
