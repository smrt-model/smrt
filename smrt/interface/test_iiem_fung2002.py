import numpy as np

from smrt.interface.iiem_fung2002 import IIEM_Fung2002
from smrt.utils import dB


def test_iem_fung92():
    eps_r = 3 + 0.1j

    iem_fung = IIEM_Fung2002(roughness_rms=0.429e-2, corr_length=3e-2)

    frequency = 2.2e9

    mu = np.cos(np.deg2rad([30, 50, 60]))

    R = iem_fung.diffuse_reflection_matrix(frequency, 1, eps_r, mu, mu, np.pi, 3)
    sigma_vv = dB(4 * np.pi * mu * R[0, 0, 0].diagonal())
    sigma_hh = dB(4 * np.pi * mu * R[1, 1, 0].diagonal())

    print(sigma_vv)
    print(sigma_hh)

    assert np.all(np.abs(sigma_vv - [-20.29314152, -24.3680021, -26.75445797]) < 1e-2)
    assert np.all(np.abs(sigma_hh - [-22.06848343, -28.6167487, -32.36807159]) < 1e-2)
