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

    res_theta = res.coords["theta"]
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
    res = m.run(sensor, sp)
    assert res.sigmaVV() == 0


def test_radar_nadir():
    # Will throw error if doesn't run
    sp = setup_2layer_snowpack()
    sensor = active(13e9, 0)
    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)
    assert res.sigmaVV() == 0


def test_radiometer_nadir():
    sp = setup_snowpack()

    theta = [0, 5]
    sensor = passive(37e9, theta)

    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)

    np.testing.assert_allclose(res.TbV(), sp.layers[0].temperature)


def test_output_stream():
    # Will throw error if doesn't run
    sp = setup_2layer_snowpack()
    sensor = active(13e9, 45)
    m = Model(NonScattering, DORT)
    res = m.run(sensor, sp)

    print(res.other_data["stream_angles"])
    assert np.allclose(res.other_data["stream_angles"], np.array([41.91460595, 45.86542465]))


def test_shallow_snowpack():
    warnings.filterwarnings("error", message=".*optically shallow.*", module=".*dort")

    with pytest.raises(SMRTWarning):
        sp = make_snowpack(
            [0.5, 0.5], "homogeneous", density=[300, 250], temperature=2 * [250], interface=2 * [Transparent]
        )
        sensor = active(13e9, 45)
        m = Model(NonScattering, DORT)
        m.run(sensor, sp, parallel_computation=False).sigmaVV()


def test_shur_based_diagonalisation():
    sp = make_snowpack(
        thickness=[1000], microstructure_model="independent_sphere", density=280, temperature=265, radius=0.05e-3
    )

    scatt = active(10e9, 50)

    m_max = 6
    nstreams = 32

    # this setting fails when DORT  use scipy.linalg.eig
    # but this works with the shur diagonalization. Let check this:

    m = Model(Rayleigh, DORT, rtsolver_options=dict(m_max=m_max, n_max_stream=nstreams, diagonalization_method="shur"))

    m.run(scatt, sp).sigmaVV()


def test_shur_forcedtriu_based_diagonalisation():
    sp = make_snowpack(
        thickness=[1000], microstructure_model="exponential", density=280, temperature=265, corr_length=0.05e-3
    )

    scatt = active(10e9, 50)

    m_max = 16
    nstreams = 32

    # this setting fails when DORT  use scipy.linalg.eig and using shur
    # but this works with the shur_forcedtriu diagonalization. Let check this:

    m = Model(
        IBA, DORT, rtsolver_options=dict(m_max=m_max, n_max_stream=nstreams, diagonalization_method="shur_forcedtriu")
    )

    m.run(scatt, sp).sigmaVV()


def test_symmetrization():
    scatt = active(10e9, 50)
    sp = make_snowpack(
        thickness=[1000], microstructure_model="exponential", density=280, temperature=265, corr_length=0.05e-3
    )

    mu = np.array([0.5, 0.2, -0.5, -0.2])
    mu = np.array([0.5, 0.2, -0.5, -0.2])

    P = IBA(scatt, sp.layers[0]).ft_even_phase(mu, mu, m_max=1).compress(mode=1)
    # print("P=", P[0:6, 0:6], "\n", P[6:, 6:])
    # print("P=", P[6:, 0:6]) #, "\n", P[0:6, 6:])

    symP = symmetrize_phase_matrix(P, m=1)

    # print("symP=", symP[0:6, 0:6], "\n", symP[6:, 6:])

    # print("symP=", symP[6:, 0:6]) #, "\n", symP[0:6, 6:])
    # P[0:]

    npt.assert_allclose(P[0:6, 0:6], symP[0:6, 0:6])
    npt.assert_allclose(P[6:, 6:], symP[6:, 6:])

    npt.assert_allclose(P[6:, 0:6], symP[6:, 0:6])
    npt.assert_allclose(P[0:6, 6:], symP[0:6, 6:])

    npt.assert_allclose(P, symP)
