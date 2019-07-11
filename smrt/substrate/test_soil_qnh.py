
import numpy as np

from smrt.inputs.make_soil import make_soil


def test_make_soil_qnh():

    make_soil('soil_qnh', 'dobson85', 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100, Q=0.16, Nv=0.11, Nh=1.63, H=0.65)

def test_make_soil_qnh_params():

    make_soil('soil_qnh', 'dobson85', 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100, H=0.65)

def soil_setup():
    s = make_soil('soil_qnh', 'dobson85', 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100, Q=0.16, Nv=0.11,
                  Nh=1.63, H=0.65)

    frequency = 1.4e9
    mu1 = np.cos(np.radians(np.arange(10, 80)))

    return s, frequency, mu1


def test_soil_qnh_reflection():

    s, frequency, mu1 = soil_setup()

    npol = 2

    refl = s.specular_reflection_matrix(frequency, 1, mu1, npol)
    print(refl)
    np.testing.assert_allclose(refl[0, 0], 0.330933, atol=1e-6)
    np.testing.assert_allclose(refl[1, 69], 0.748025, atol=1e-6)


def test_soil_qnh_emissivity():

    s, frequency, mu1 = soil_setup()

    npol = 2

    abso = s.emissivity_matrix(frequency, 1, mu1, npol)
    print(abso)
    np.testing.assert_allclose(abso[0, 0], 0.669067, atol=1e-6)
    np.testing.assert_allclose(abso[1, 69], 0.251975, atol=1e-6)