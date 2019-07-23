
import numpy as np


from smrt.inputs.make_soil import make_soil


def test_make_soil_wegmuller():

    make_soil('soil_wegmuller', 'dobson85', 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100, roughness_rms=1e-2)


def test_soil_wegmuller_reflection():

    s = make_soil('soil_wegmuller', 'dobson85', 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100, roughness_rms=1e-2)

    frequency = 37e9
    mu1 = np.cos(np.radians(np.arange(10, 80)))

    npol = 2

    s.specular_reflection_matrix(frequency, 1, mu1, npol)
