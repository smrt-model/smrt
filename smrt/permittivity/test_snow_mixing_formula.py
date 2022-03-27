
import numpy as np

from smrt.core.globalconstants import DENSITY_OF_ICE, DENSITY_OF_WATER
from .snow_mixing_formula import wetsnow_permittivity_hallikainen86


def test_wetsnow_permittivity_hallikainen86():

    fw = 0.1  # 10 percent water per snow volume
    dry_density_gm3 = 0.24
    snow_density = 1000 * ((1 - fw) * dry_density_gm3 + fw * 1)  # 1 g/cm3

    fi = (snow_density - fw * DENSITY_OF_WATER) / DENSITY_OF_ICE
    liquid_water = fw / (fw + fi)

    # Data from the graph in Fig 7
    frequency = 6e9
    eps = wetsnow_permittivity_hallikainen86(frequency, snow_density, liquid_water)

    print("eps:", eps, eps - 1 - 1.83 * dry_density_gm3)
    assert np.allclose((eps - 1 - 1.83 * dry_density_gm3).real, 1.24, atol=0.01)
    assert np.allclose(eps.imag, 0.685, atol=0.001)

    # Data from the graph in Fig 8
    # the incremental epsilon for the modified debye-model seems defined with respect to A = 1 + 1.83 * dry_snow_density_gcm3 + B1
    frequency = 37e9
    eps = wetsnow_permittivity_hallikainen86(frequency, snow_density, liquid_water)

    print("eps:", eps, eps - 1 - 1.83 * dry_density_gm3)
    assert np.allclose((eps - 1 - 1.83 * dry_density_gm3).real, 0.319, atol=0.01)
    assert np.allclose(eps.imag, 0.468, atol=0.001)
