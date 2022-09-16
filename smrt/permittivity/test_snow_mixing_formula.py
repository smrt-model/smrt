
import pytest
import numpy as np

from smrt.core.globalconstants import DENSITY_OF_ICE, DENSITY_OF_WATER
from .snow_mixing_formula import wetsnow_permittivity_hallikainen86, \
    wetsnow_permittivity_hallikainen86_ulaby14, \
    wetsnow_permittivity_colbeck80_caseI, wetsnow_permittivity_colbeck80_caseII, wetsnow_permittivity_colbeck80_caseIII


@pytest.mark.skip(reason="no way to obtain an agreement between the equations and the graphs in H86")
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


def test_wetsnow_permittivity_hallikainen86_ulaby14():

    # values from fig 4-22 and 4-23. (freq,mv) : eps
    a_val = {
        (3e9, 0.02): 1.604 + 0.051j,
        (37e9, 0.02): 1.303 + 0.056j,
        (3e9, 0.06): 2.129 + 0.218j,
        (37e9, 0.06): 1.43 + 0.239j,
        (3e9, 0.1): 2.769 + 0.427j,
        (37e9, 0.1): 1.567 + 0.468j,
    }

    dry_density_gm3 = 0.25

    for k, v in a_val.items():

        fw = k[1]  # 10 percent water per snow volume

        snow_density = 1000 * ((1 - fw) * dry_density_gm3 + fw * 1)  # 1 g/cm3

        fi = (snow_density - fw * DENSITY_OF_WATER) / DENSITY_OF_ICE
        liquid_water = fw / (fw + fi)

        # Data from the graph in Fig 7
        frequency = k[0]
        eps = wetsnow_permittivity_hallikainen86_ulaby14(frequency, snow_density, liquid_water)

        print("eps:", eps)
        print("reference:", v)
        assert np.allclose(eps.real, v.real, atol=0.001)
        assert np.allclose(eps.imag, v.imag, atol=0.001)


def test_wetsnow_permittivity_colbeck80_caseI():

    # values from Fig 5
    v = 1.891
    frequency = 1e9
    temperature = 273.15
    density = DENSITY_OF_ICE / 2

    d = wetsnow_permittivity_colbeck80_caseI(frequency=frequency, temperature=temperature, density=density, liquid_water=0).real
    print("dilectric constant: ", d)
    print("reference: ", v)

    assert np.allclose(d, v, atol=0.001)


def test_wetsnow_permittivity_colbeck80_caseII():

    DENSITY_OF_AIR = 1.22

    # values from Fig 9
    v = 3.2
    frequency = 1e9
    temperature = 273.15

    theta_w = 0.10
    phi = 0.4
    theta_a = phi - theta_w
    theta_i = 1 - phi
    density = theta_w * DENSITY_OF_WATER + theta_a * DENSITY_OF_AIR + theta_i * DENSITY_OF_ICE

    lwc = theta_w / (theta_w + theta_i)

    d = wetsnow_permittivity_colbeck80_caseII(frequency=frequency, temperature=temperature, density=density, liquid_water=lwc).real

    print("dilectric constant: ", d)
    print("reference: ", v)

    assert np.allclose(d, v, atol=0.001)


def test_wetsnow_permittivity_colbeck80_caseIII():

    DENSITY_OF_AIR = 1.22

    # values from Fig 8
    v = 5.062
    frequency = 1e9
    temperature = 273.15

    theta_w = 0.16
    phi = 0.32
    theta_a = phi - theta_w
    theta_i = 1 - phi

    density_d = phi * DENSITY_OF_AIR + theta_i * DENSITY_OF_ICE
    density_s = theta_w * DENSITY_OF_WATER + theta_a * DENSITY_OF_AIR + theta_i * DENSITY_OF_ICE

    lwc = theta_w / (theta_w + theta_i)

    d_d = wetsnow_permittivity_colbeck80_caseI(frequency=frequency, temperature=temperature, density=density_d, liquid_water=0).real
    d_s = wetsnow_permittivity_colbeck80_caseIII(frequency=frequency, temperature=temperature, density=density_s, liquid_water=lwc).real

    print("delta dilectric constant: ", d_s - d_d)
    print("reference: ", v)

    assert np.allclose(d_s - d_d, v, atol=0.001)
