
import numpy as np

from smrt.interface.geometrical_optics import GeometricalOptics
from smrt.interface.geometrical_optics_backscatter import GeometricalOpticsBackscatter


def test_compare_geometrical_optics():
    # check that both code for geometrical optics gives the same backscatter

    eps_2 = 40 + 3j
    eps_1 = 1

    ks = 2.8
    kl = 7.5

    go = GeometricalOptics(mean_square_slope=2 * ks**2 / kl**2)
    go_back = GeometricalOpticsBackscatter(mean_square_slope=2 * ks**2 / kl**2)

    theta_i = np.arange(1, 80)

    mu_i = np.cos(np.deg2rad(theta_i))
    mu_s = mu_i
    dphi = np.pi

    m = go.diffuse_reflection_matrix(10e9, eps_1, eps_2, mu_s, mu_i, dphi, 2)
    m = m.diagonal

    m_back = go_back.diffuse_reflection_matrix(10e9, eps_1, eps_2, mu_s, mu_i, dphi, 2)
    m_back = m_back.diagonal

    #print(m[0], m_back[0])
    assert np.allclose(m[0], m_back[0])
    assert np.allclose(m[1], m_back[1])


def test_reflectance_reciprocity():
    eps_1 = 1
    eps_2 = 1.6
    mean_square_slope = 0.5

    go = GeometricalOptics(mean_square_slope=mean_square_slope, shadow_correction=False)

    dphi = np.linspace(0, 2 * np.pi, 10)

    for mu_i in np.linspace(0.1, 1, 10):
        for mu_s in np.linspace(0.1, 1, 10):

            R = go.diffuse_reflection_matrix(10e9, eps_1, eps_2, mu_s, mu_i, dphi, 2).values
            Rs = go.diffuse_reflection_matrix(10e9, eps_1, eps_2, mu_i, mu_s, dphi, 2).values

            assert np.allclose(R[1, 0, :] * mu_i, Rs[0, 1, :] * mu_s, atol=1e-3)
            assert np.allclose(R[0, 0, :] * mu_i, Rs[0, 0, :] * mu_s, atol=1e-3)
            assert np.allclose(R[1, 1, :] * mu_i, Rs[1, 1, :] * mu_s, atol=1e-3)


def test_transmission_reciprocity():
    eps_1 = 1
    eps_2 = 1.6
    mean_square_slope = 0.5

    go = GeometricalOptics(mean_square_slope=mean_square_slope, shadow_correction=False)

    dphi = np.linspace(0, 2 * np.pi, 10)

    for mu_i in np.linspace(0.1, 1, 10):
        for mu_t in np.linspace(0.1, 1, 10):
            T = go.diffuse_transmission_matrix(10e9, eps_1, eps_2, mu_t, mu_i, dphi, 2).values
            Tt = go.diffuse_transmission_matrix(10e9, eps_2, eps_1, mu_i, mu_t, dphi, 2).values

            assert np.allclose(T[1, 0, :] * mu_i * eps_1, Tt[0, 1, :] * mu_t * eps_2, atol=1e-3)
            assert np.allclose(T[0, 0, :] * mu_i * eps_1, Tt[0, 0, :] * mu_t * eps_2, atol=1e-3)
            assert np.allclose(T[1, 1, :] * mu_i * eps_1, Tt[1, 1, :] * mu_t * eps_2, atol=1e-3)
