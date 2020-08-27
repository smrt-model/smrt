
import numpy as np
from smrt.interface.iem_fung92_brogioni10 import IEM_Fung92_Briogoni10
from smrt.interface.iem_fung92 import IEM_Fung92
from smrt.utils import dB

def test_iem_fung92_biogoni10_continuty():

    eps_r = 3 + 0.1j

    iem_fung = IEM_Fung92(roughness_rms=0.429e-2, corr_length=3e-2)
    iem_fung_brogioni = IEM_Fung92_Briogoni10(roughness_rms=0.429e-2, corr_length=3e-2)

    frequency = 2.2e9

    mu = np.cos(np.deg2rad([30, 50, 60]))

    R = iem_fung.diffuse_reflection_matrix(frequency, 1, eps_r, mu, mu, np.pi, 2, debug=True)

    R2 = iem_fung_brogioni.diffuse_reflection_matrix(frequency, 1, eps_r, mu, mu, np.pi, 2, debug=True)
    
    sigma_vv = dB(4*np.pi * mu * R[0])
    sigma_hh = dB(4*np.pi * mu * R[1])

    sigma_vv2 = dB(4*np.pi * mu * R2[0])
    sigma_hh2 = dB(4*np.pi * mu * R2[1])

    assert np.allclose(sigma_vv, sigma_vv2)
    assert np.allclose(sigma_hh, sigma_hh2)



def test_iem_fung92_brogioni10():

    eps_r = 3 + 0.1j

    iem_fung = IEM_Fung92_Briogoni10(roughness_rms=0.429e-2, corr_length=30e-2)

    frequency = 2.2e9

    mu = np.cos(np.deg2rad([30, 50, 60]))

    R = iem_fung.diffuse_reflection_matrix(frequency, 1, eps_r, mu, mu, np.pi, 2, debug=True)
    sigma_vv = dB(4*np.pi * mu * R[0])
    sigma_hh = dB(4*np.pi * mu * R[1])

    print(sigma_vv)
    print(sigma_hh)

    assert np.all(np.abs(sigma_vv - [-25.8475821, -28.09794986, -27.1320767 ]) < 1e-2)
    assert np.all(np.abs(sigma_hh - [-31.30415086, -40.67474292, -29.06341978]) < 1e-2)
