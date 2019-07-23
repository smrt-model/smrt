
import numpy as np

from smrt.interface.geometrical_optics import GeometricalOptics
from smrt.interface.geometrical_optics_backscatter import GeometricalOpticsBackscatter


def test_compare_geometrical_optics():
    # check that both code for geometrical optics gives the same backscatter

    eps_2 = 40 + 3j
    eps_1 = 1

    ks = 2.8
    kl = 7.5

    go = GeometricalOptics(mean_square_slope = 2*ks**2/kl**2)
    go_back = GeometricalOpticsBackscatter(mean_square_slope = 2*ks**2/kl**2)


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
