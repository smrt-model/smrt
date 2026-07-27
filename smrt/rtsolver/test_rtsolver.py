import warnings

import numpy as np
import numpy.testing as npt
import pytest

from smrt import make_snowpack
from smrt.core.error import SMRTWarning
from smrt.core.model import Model
from smrt.core.sensor import active, passive
from smrt.emmodel.nonscattering import NonScattering
from smrt.interface.transparent import Transparent
from smrt.rtsolver.dort import DORT


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


@pytest.mark.parametrize("rtsolver", ["dort", "successive_order"])
def test_noabsoprtion(setup_snowpack, rtsolver):
    sp = setup_snowpack

    sensor = passive(37e9, theta=[30, 40])
    m = Model(NonScattering, rtsolver)
    res = m.run(sensor, sp)

    tb = res.TbV(order="total") if "order" in res.coords else res.TbV()
    np.testing.assert_allclose(tb, sp.layers[0].temperature, atol=0.01)


@pytest.mark.parametrize("rtsolver", ["dort", "successive_order"])
def test_returned_theta(setup_snowpack, rtsolver):
    sp = setup_snowpack

    theta = [30, 40]
    sensor = passive(37e9, theta)

    m = Model(NonScattering, rtsolver)
    res = m.run(sensor, sp)

    res_theta = res.coords["theta"]
    print(res_theta)
    np.testing.assert_allclose(res_theta, theta)


@pytest.mark.parametrize("rtsolver", ["dort", "successive_order"])
def test_output_stream(setup_2layer_snowpack, rtsolver):
    # Will throw error if doesn't run
    sp = setup_2layer_snowpack
    sensor = active(13e9, 45)
    m = Model(NonScattering, rtsolver)
    res = m.run(sensor, sp)

    print(res.other_data["stream_angles"])

    np.testing.assert_allclose(res.other_data["stream_angles"], np.array([41.91460595, 45.86542465]))


@pytest.mark.parametrize("angle", [(45), (0)])
@pytest.mark.parametrize("rtsolver", ["dort", "successive_order"])
def test_depth_hoar(setup_snowpack_with_DH, angle, rtsolver):
    # Will throw error if doesn't run
    sp = setup_snowpack_with_DH
    sensor = active(13e9, angle)
    m = Model(NonScattering, rtsolver)
    m.run(sensor, sp).sigmaVV()


@pytest.mark.parametrize("angle", [(45), (0)])
@pytest.mark.parametrize("rtsolver", ["dort", "successive_order"])
def test_2layer_pack(setup_2layer_snowpack, angle, rtsolver):
    # Will throw error if doesn't run
    sp = setup_2layer_snowpack
    sensor = active(13e9, angle)
    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)
    assert res.sigmaVV() == 0


@pytest.mark.parametrize("rtsolver", ["dort", "successive_order"])
def test_radiometer_nadir(setup_snowpack, rtsolver):
    sp = setup_snowpack

    theta = [0, 5]
    sensor = passive(37e9, theta)
    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)
    np.testing.assert_allclose(res.TbV(), sp.layers[0].temperature)


@pytest.mark.parametrize("rtsolver", ["dort"])
def test_shallow_snowpack(rtsolver):
    warnings.filterwarnings("error", message=".*optically shallow.*", module=".*dort")

    with pytest.raises(SMRTWarning):
        sp = make_snowpack(
            [0.5, 0.5], "homogeneous", density=[300, 250], temperature=2 * [250], interface=2 * [Transparent]
        )
        sensor = active(13e9, 45)
        m = Model(NonScattering, rtsolver)
        m.run(sensor, sp, parallel_computation=False).sigmaVV()


@pytest.mark.parametrize("rtsolver", ["dort", "successive_order"])
def test_rayleigh_jeans_approximation(setup_snowpack, rtsolver):
    sp = setup_snowpack

    theta = [30, 40]
    sensor = passive(300e9, theta)

    m = Model(NonScattering, DORT, rtsolver_options=dict(rayleigh_jeans_approximation=True))
    res_rj = m.run(sensor, sp)

    m = Model(NonScattering, DORT, rtsolver_options=dict(rayleigh_jeans_approximation=False))
    res_full = m.run(sensor, sp)

    # at 300GHz and 250K, the RJ approximation is not very accurate
    print(res_rj.TbV(), res_full.TbV())
    npt.assert_allclose(res_rj.data, res_full.data, rtol=0.01)
