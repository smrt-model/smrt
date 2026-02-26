# coding: utf-8
"""
Provide equations to compute the effective permittivity of water.
"""

import numpy as np

from ..core.error import SMRTError
from ..core.globalconstants import FREEZING_POINT, GHz
from ..core.layer import layer_properties


@layer_properties("temperature")
def water_permittivity_maetzler87(frequency, temperature):
    """
    Calculate the complex water dielectric constant depending on the frequency and temperature based on Mätzler & Wegmuller (1987).

    Args:
        frequency: Frequency in Hz.
        temperature: Temperature in K.

    Raises:
        Exception: If liquid water > 0 or salinity > 0 (model unsuitable).

    Returns:
        complex: Complex permittivity of pure ice.

    References:
        Matzler, C., Wegmuller, U., (1987) Dielectric properties of freshwater ice at microwave frequencies. J. Phys. D:
        Appl. Phys, 20, 1623–1630, https://doi.org/10.1088/0022-3727/20/12/013.
    """
    if temperature < FREEZING_POINT:
        raise SMRTError(f"The water temperature must be higher or equal to {FREEZING_POINT}K")

    freqGHz = frequency / 1e9

    theta = 1 - 300.0 / temperature

    e0 = 77.66 - 103.3 * theta
    e1 = 0.0671 * e0

    f1 = 20.2 + 146.4 * theta + 316 * theta**2
    e2 = 3.52 + 7.52 * theta
    #  % version of Liebe MPM 1993 uses: e2=3.52
    f2 = 39.8 * f1

    Ew = e2 + (e1 - e2) / complex(1, -freqGHz / f2) + (e0 - e1) / complex(1, -freqGHz / f1)

    return Ew


# for backward compatibility
water_permittivity = water_permittivity_maetzler87


@layer_properties("temperature")
def water_permittivity_tiuri80(frequency, temperature):
    """
    Calculate the complex water dielectric constant reported by Tiuri and Schultz (1980).

    Args:
        frequency: Frequency in Hz.
        temperature: Temperature in K.

    Raises:
        SMRTError: If the water temperature is below the freezing point.

    Returns:
        complex: Complex permittivity of water.

    References:
        Tiuri, M. and Schultz, H., Theoretical and experimental studies of microwave radiation from a natural snow field. In Rango, A. , ed.
        Microwave remote sensing of snowpack properties. Proceedings of a workshop ... Fort Collins, Colorado, May 20-22, 1980.
        Washington, DC, National Aeronautics and Space Center, 225-234. (Conference Publication 2153.)
        https://ntrs.nasa.gov/api/citations/19810010984/downloads/19810010984.pdf
    """
    freqGHz = frequency / GHz

    tempC = temperature - FREEZING_POINT

    if np.any(tempC < 0):
        raise SMRTError(f"The water temperature must be higher or equal to {FREEZING_POINT}K")

    e2 = 4.903e-2

    e1 = 87.74 - 0.4008 * tempC + 9.398e-4 * tempC**2 + 1.410e-6 * tempC**3

    # version of Liebe 1991 because Tiuri 1980 does not give the relaxation frequency
    theta = 1 - 300.0 / temperature
    f1 = 20.2 + 146.4 * theta + 316 * theta**2

    Ew = e2 + (e1 - e2) / complex(1, -freqGHz / f1)

    return Ew


def water_permittivity_turner16(frequency, temperature):
    """
    Calculate the complex water dielectric constant depending on the frequency and temperature based on Turner et al.
    (2016).

    Args:
        frequency: Frequency in Hz. temperature: Temperature in K.

    References:

    Turner, D. D., Kneifel, S., & Cadeddu, M. P. (2016). An Improved Liquid Water Absorption Model at Microwave
    Frequencies for Supercooled Liquid Water Clouds. Journal of Atmospheric and Oceanic Technology, 33(1), 33–44.
    https://doi.org/10.1175/jtech-d-15-0074.1
    """

    # TKC coefficients
    a1 = 8.111e1
    b1 = 4.434e-03
    c1 = 1.302e-13
    d1 = 6.627e2

    a2 = 2.025e0
    b2 = 1.073e-02
    c2 = 1.012e-14
    d2 = 6.089e2
    t_c = 1.342e2

    # Static dielectric constant coefficients (Eq. 6)  Hamelin et al. 1998
    s0 = 8.79144e01
    s1 = -4.04399e-01
    s2 = 9.58726e-04
    s3 = -1.32802e-06

    tempC = temperature - FREEZING_POINT

    eps_s = s0 + s1 * tempC + s2 * tempC**2 + s3 * tempC**3  # Eq. 6:

    tau1 = debye_tau_i(c1, d1, tempC, t_c)
    delta1 = debye_delta_i(a1, b1, tempC)
    A1 = debye_A_i(tau1, delta1, frequency)

    tau2 = debye_tau_i(c2, d2, tempC, t_c)
    delta2 = debye_delta_i(a2, b2, tempC)
    A2 = debye_A_i(tau2, delta2, frequency)

    print(A1, A2)

    eps_real = eps_s - (2 * np.pi * frequency) ** 2 * (A1 + A2)  # eq 4

    B1 = debye_B_i(tau1, delta1, frequency)
    B2 = debye_B_i(tau2, delta2, frequency)
    eps_imag = (2 * np.pi * frequency) * (B1 + B2)  # eq 5

    return eps_real + 1j * eps_imag


def debye_delta_i(a_i, b_i, tempC):
    """Compute Delta_i as a function of temperature (T in °C). Eq. 9 in Turner 2016"""
    return a_i * np.exp(-b_i * tempC)


def debye_tau_i(c_i, d_i, tempC, t_c):
    """Compute Tau_i as a function of temperature (T in °C). Eq. 10 in Turner 2016"""
    return c_i * np.exp(d_i / (tempC + t_c))


def debye_A_i(tau_i, delta_i, frequency):
    """Compute A_i relaxation term. Eq. 7 in Turner 2016"""
    return (tau_i**2 * delta_i) / (1 + (2 * np.pi * frequency * tau_i) ** 2)


def debye_B_i(tau_i, delta_i, frequency):
    """Compute B_i relaxation term. Eq. 8 in Turner 2016"""
    return (tau_i * delta_i) / (1 + (2 * np.pi * frequency * tau_i) ** 2)
