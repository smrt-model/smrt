import numpy as np
import pytest

from smrt.interface.geometrical_optics import GeometricalOptics
from smrt.interface.geometrical_optics_backscatter import GeometricalOpticsBackscatter


def get_diffuse_reflection(go):
    eps_2 = 40 + 3j
    eps_1 = 1

    freq = 10e9

    theta_i = np.arange(1, 80)

    mu_i = np.cos(np.deg2rad(theta_i))
    mu_s = mu_i
    dphi = np.pi

    m = go.diffuse_reflection_matrix(freq, eps_1, eps_2, mu_s, mu_i, dphi, 2)
    return m.diagonal


def test_compare_geometrical_optics():
    # check that both code for geometrical optics gives the same backscatter

    ks = 2.8
    kl = 7.5

    go = GeometricalOptics(mean_square_slope=2 * ks**2 / kl**2)
    go_back = GeometricalOpticsBackscatter(mean_square_slope=2 * ks**2 / kl**2)

    m = get_diffuse_reflection(go)
    m_back = get_diffuse_reflection(go_back)

    # print(m[0], m_back[0])
    np.testing.assert_allclose(m[0, 0], m_back[0])
    np.testing.assert_allclose(m[1, 0], m_back[1])


@pytest.mark.parametrize("interface", [(GeometricalOptics), (GeometricalOpticsBackscatter)])
def test_parameters_geometrical_optics_and_backscatter(interface):
    s = 2.8e-2
    l = 7.5e-2

    go_mss = interface(mean_square_slope=2 * s**2 / l**2)
    go_rms_corr = interface(roughness_rms=s, corr_length=l)

    m_mss = get_diffuse_reflection(go_mss)
    m_rms_corr = get_diffuse_reflection(go_rms_corr)

    np.testing.assert_allclose(m_mss[0], m_rms_corr[0])
    np.testing.assert_allclose(m_mss[1], m_rms_corr[1])


# The two following tests seem difficult to factorise
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

            np.testing.assert_allclose(R[1, 0, :] * mu_i, Rs[0, 1, :] * mu_s, atol=1e-3)
            np.testing.assert_allclose(R[0, 0, :] * mu_i, Rs[0, 0, :] * mu_s, atol=1e-3)
            np.testing.assert_allclose(R[1, 1, :] * mu_i, Rs[1, 1, :] * mu_s, atol=1e-3)


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

            np.testing.assert_allclose(T[1, 0, :] * mu_i * eps_1, Tt[0, 1, :] * mu_t * eps_2, atol=1e-3)
            np.testing.assert_allclose(T[0, 0, :] * mu_i * eps_1, Tt[0, 0, :] * mu_t * eps_2, atol=1e-3)
            np.testing.assert_allclose(T[1, 1, :] * mu_i * eps_1, Tt[1, 1, :] * mu_t * eps_2, atol=1e-3)
