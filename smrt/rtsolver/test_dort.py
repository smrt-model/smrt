import warnings

import numpy as np
import numpy.testing as npt
import pytest

from smrt import make_snowpack
from smrt.core.error import SMRTWarning
from smrt.core.model import Model
from smrt.core.sensor import active, passive
from smrt.emmodel.iba import IBA
from smrt.emmodel.nonscattering import NonScattering
from smrt.emmodel.rayleigh import Rayleigh
from smrt.interface.transparent import Transparent
from smrt.rtsolver.dort import DORT, symmetrize_phase_matrix


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


def test_noabsoprtion(setup_snowpack):
    sp = setup_snowpack

    sensor = passive(37e9, theta=[30, 40])
    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)

    np.testing.assert_allclose(res.data, sp.layers[0].temperature)


def test_returned_theta(setup_snowpack):
    sp = setup_snowpack

    theta = [30, 40]
    sensor = passive(37e9, theta)

    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)

    res_theta = res.coords["theta"]
    print(res_theta)
    np.testing.assert_allclose(res_theta, theta)


def test_selectby_theta(setup_snowpack):
    sp = setup_snowpack

    theta = [30, 40]
    sensor = passive(37e9, theta)

    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)

    print(res.data.coords)
    res.TbV(theta=30)


def test_depth_hoar_stream_numbers(setup_snowpack_with_DH):
    # Will throw error if doesn't run
    sp = setup_snowpack_with_DH
    sensor = active(13e9, 45)
    m = Model(NonScattering, DORT)
    m.run(sensor, sp).sigmaVV()


@pytest.mark.parametrize("angle", [(45), (0)])
def test_2layer_pack(setup_2layer_snowpack, angle):
    # Will throw error if doesn't run
    sp = setup_2layer_snowpack
    sensor = active(13e9, angle)
    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)
    assert res.sigmaVV() == 0


def test_radiometer_nadir(setup_snowpack):
    sp = setup_snowpack

    theta = [0, 5]
    sensor = passive(37e9, theta)

    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)

    np.testing.assert_allclose(res.TbV(), sp.layers[0].temperature)


def test_output_stream(setup_2layer_snowpack):
    # Will throw error if doesn't run
    sp = setup_2layer_snowpack
    sensor = active(13e9, 45)
    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)

    print(res.other_data["stream_angles"])
    np.testing.assert_allclose(res.other_data["stream_angles"], np.array([41.91460595, 45.86542465]))


def test_shallow_snowpack():
    warnings.filterwarnings("error", message=".*optically shallow.*", module=".*dort")

    with pytest.raises(SMRTWarning):
        sp = make_snowpack(
            [0.5, 0.5], "homogeneous", density=[300, 250], temperature=2 * [250], interface=2 * [Transparent]
        )
        sensor = active(13e9, 45)
        m = Model(NonScattering, DORT)
        m.run(sensor, sp, parallel_computation=False).sigmaVV()


@pytest.mark.parametrize(
    "microstructure_model,m_max,emmodel,diagonalization_method",
    [("independent_sphere", 6, Rayleigh, "shur"), ("exponential", 16, IBA, "shur_forcedtriu")],
)
def test_shur_based_diagonalisation(microstructure_model, m_max, emmodel, diagonalization_method):
    sp = make_snowpack(
        thickness=[1000],
        microstructure_model=microstructure_model,
        density=280,
        temperature=265,
        radius=0.05e-3,
        corr_length=0.05e-3,
    )
    scatt = active(10e9, 50)
    nstreams = 32

    # this setting fails when DORT  use scipy.linalg.eig
    # but this works with the shur diagonalization. Let check this:

    m = Model(
        emmodel,
        DORT,
        rtsolver_options=dict(m_max=m_max, n_max_stream=nstreams, diagonalization_method=diagonalization_method),
    )

    m.run(scatt, sp).sigmaVV()


def test_symmetrization():
    scatt = active(10e9, 50)
    sp = make_snowpack(
        thickness=[1000], microstructure_model="exponential", density=280, temperature=265, corr_length=0.05e-3
    )

    mu = np.array([0.5, 0.2, -0.5, -0.2])

    P = IBA(scatt, sp.layers[0]).ft_even_phase(mu, mu, m_max=1).compress(mode=1)

    symP = symmetrize_phase_matrix(P, m=1)

    npt.assert_allclose(P[0:6, 0:6], symP[0:6, 0:6])
    npt.assert_allclose(P[6:, 6:], symP[6:, 6:])

    npt.assert_allclose(P[6:, 0:6], symP[6:, 0:6])
    npt.assert_allclose(P[0:6, 6:], symP[0:6, 6:])

    npt.assert_allclose(P, symP)


def test_rayleigh_jeans_approximation(setup_snowpack):
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
