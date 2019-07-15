
import numpy as np
from smrt.interface.iem_fung92 import IEM_Fung92
from smrt.utils import dB

def test_iem_fung92():

    eps_r = 3 + 0.1j

    iem_fung = IEM_Fung92(roughness_rms=0.429e-2, corr_length=3e-2)

    frequency = 2.2e9

    mu = np.cos(np.deg2rad([30, 50, 60]))

    R = iem_fung.diffuse_reflection_matrix(frequency, 1, eps_r, mu, mu, np.pi, 2, debug=True)
    sigma_vv = dB(4*np.pi * mu * R[0])
    sigma_hh = dB(4*np.pi * mu * R[1])

    print(sigma_vv)
    print(sigma_hh)

    assert np.all(np.abs(sigma_vv - [-20.25297061, -24.35232625, -26.74346526]) < 1e-2)
    assert np.all(np.abs(sigma_hh - [-22.10327899, -28.69367149, -32.53013663]) < 1e-2)
