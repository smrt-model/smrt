# coding: utf-8

import numpy as np

from smrt.core.globalconstants import FREEZING_POINT, GHz, PERMITTIVITY_OF_FREE_SPACE, PSU
from .brine import brine_conductivity, brine_relaxation_time, permittivity_high_frequency_limit, \
    static_brine_permittivity
from ..core.layer import layer_properties


@layer_properties("temperature", "salinity")
def seawater_permittivity_klein76(frequency, temperature, salinity):
    """Calculates permittivity (dielectric constant) of water using an empirical relationship described
       by Klein and Swift (1976).

       :param frequency: frequency in Hz
       :param temperature: water temperature in K
       :param salinity: water salinity in kg/kg (see PSU constant in smrt module)
       Returns complex water permittivity for a frequency f."""

    tempC = temperature - FREEZING_POINT
    Sppt = salinity / PSU

    omega = 2 * np.pi * frequency
    eps_inf = 4.9  # limiting high frequency value

    # calculate static dielectric constant of saline water:
    eps_s_T = 87.134 - 1.949e-1 * tempC - 1.276e-2 * tempC ** 2 + 2.491e-4 * tempC ** 3
    a_ST = 1. + 1.613e-5 * Sppt * tempC - 3.656e-3 * Sppt + 3.210e-5 * Sppt ** 2 - 4.232e-7 * Sppt ** 3
    eps_static = eps_s_T * a_ST

    # calculate tau = relaxation time of saline water:
    tau_T0 = 1.768e-11 - 6.086e-13 * tempC + 1.104e-14 * tempC ** 2 - 8.111e-17 * tempC ** 3
    b_ST = 1. + 2.282e-5 * Sppt * tempC - 7.638e-4 * Sppt - 7.760e-6 * Sppt ** 2 + 1.105e-8 * Sppt ** 3
    tau = tau_T0 * b_ST

    # calculate sigma = ionic conductivity of dissolved salts:
    delta = 25 - tempC
    beta = 2.0333e-2 + 1.266e-4 * delta + 2.464e-6 * delta ** 2 - Sppt * \
           (1.849e-5 - 2.551e-7 * delta + 2.551e-8 * delta ** 2)
    sigma_25S = Sppt * (0.182521 - 1.46192e-3 * Sppt + 2.09324e-5 * Sppt ** 2 - 1.28205e-7 * Sppt ** 3)
    sigma = sigma_25S * np.exp(-delta * beta)

    # Debye type relaxation equation:
    # Similar equation form as in saline_ice.py: brine_permittivity_stogryn85
    eps_water = eps_inf + (eps_static - eps_inf) / (1 - 1j *
                                                    omega * tau) + 1j * sigma / (omega * PERMITTIVITY_OF_FREE_SPACE)
    return eps_water


@layer_properties("temperature")
def seawater_permittivity_stogryn71(frequency, temperature):
    """Computes dielectric constant of brine, complex_b (Stogryn, 1971 approach)

    Input parameters: from polynomial fit in Stogryn and Desargent, 1985

    Source: Matlab code, Ludovic Brucker

    :param frequency: frequency in Hz
    :param temperature: water temperature in K
    Returns complex water permittivity for a frequency f."""

    # High-frequency dielectric constant of saline water
    eps_inf = permittivity_high_frequency_limit(temperature)

    # Static dielectric constant of saline water
    eps_static = static_brine_permittivity(temperature)

    # Angular frequency
    omega_brine = 2 * np.pi * frequency

    # Relaxation time
    tau_brine = brine_relaxation_time(temperature)

    # Ionic conductivity
    sigma_brine = brine_conductivity(temperature)

    # Output dielectric constant of brine using Stogryn, 1971 formulation (Ulaby et al., 1986 p. 2046)
    freqGHz = frequency / GHz
    real_brine = eps_inf + ((eps_static - eps_inf) / (1 + (tau_brine * freqGHz) ** 2))
    imag_brine = (tau_brine * freqGHz) * ((eps_static - eps_inf) / (1 + (tau_brine * freqGHz) ** 2)) + (
            sigma_brine / (omega_brine * PERMITTIVITY_OF_FREE_SPACE))

    return real_brine + 1j * imag_brine


@layer_properties("temperature")
def brine_permittivity_stogryn85(frequency, temperature):
    """computes permittivity and loss of brine using equations given in Stogryn and Desargant (1985): 'The Dielectric Properties of Brine in Sea Ice at Microwave Frequencies', IEEE.

    :param frequency: em frequency [Hz]
    :param temperature: ice temperature in K"""

    eps_static = static_brine_permittivity(temperature)  # limiting static permittivity
    tau = brine_relaxation_time(temperature)  # relaxation time
    sigma = brine_conductivity(temperature)  # ionic conductivity of dissolved salts
    eps_inf = permittivity_high_frequency_limit(temperature)  # limiting high frequency value
    brine_permittivity = eps_inf + (eps_static - eps_inf) / (1. - tau * frequency / GHz *
                                                             1j) + sigma / (
                                 2. * np.pi * PERMITTIVITY_OF_FREE_SPACE * frequency) * 1j
    return brine_permittivity


@layer_properties("temperature", "salinity")
def seawater_permittivity_stogryn95(frequency, temperature, salinity):
    """Computes seawater dielectric constant using Stogryn 1995.

    source: Stogryn 1995 + http://rime.aos.wisc.edu/MW/models/src/eps_sea_stogryn.f90; Matlab code, Ludovic Brucker

    :param frequency: frequency in Hz
    :param temperature: water temperature in K
    :param salinity: water salinity in kg/kg (see PSU constant in smrt module)
    Returns complex water permittivity for a frequency f."""

    #   real, intent (in) :: f     ! [GHz]    Frequency (valid from 0 to 1000 GHz)
    #   real, intent (in) :: T     ! [°C]     Temperature
    #   real, intent (in) :: S     ! [permil] Salinity
    #   complex :: eps             !          Dielectric constant
    #   real :: eps_inf            !          High-frequency dielectric constant
    #   real :: eps_s              !          Static dielectric constant
    #   real :: eps_s0             !          Static dielectric for S=0
    #   real :: eps1               !          Intermediate dielectric
    #   real :: eps10              !          Intermediate dielectric for S=0
    #   real :: tau1x2pi           ! [ns]     First relaxation time
    #   real :: tau10x2pi          ! [ns]     First relaxation time for S=0
    #   real :: tau2x2pi           ! [ns]     Second relaxation time
    #   real :: sigma              ! [S/m]    Conductivity of sea water
    #   real :: sigma35            ! [S/m]    Conductivity for S=35
    #   real :: R15                !          Ratio of conductivity to conductivity of standard sea water at T = 15°C
    #   real :: RtR15              !          Ratio of conductivity to conductivity of standard sea water at temp. T and salinity S
    #   real :: alpha0, alpha1     !          Fitting parameters for RtR15
    #   real :: a, b, b1, b2       !          Debye parameters

    freqGHz = frequency / GHz
    Sppt = salinity / PSU
    tempC = temperature - FREEZING_POINT

    # Eq.6
    eps_s0 = (3.70886e4 - 8.2168e1 * tempC) / (4.21854e2 + tempC)

    # Eq. 8
    tau10x2pi = (255.04 + 0.7246 * tempC) / ((49.25 + tempC) * (45 + tempC))

    # Eq. 9
    tau2x2pi = 0.628e-2

    # Eq. 10
    eps_inf = 4.05 + 1.86e-2 * tempC

    # Eq. 15
    sigma35 = 2.903602 + 8.60700e-2 * tempC + 4.738817e-4 * tempC ** 2 - 2.9910e-6 * tempC ** 3 + 4.3047e-9 * tempC ** 4

    # Eq. 16
    R15 = Sppt * (37.5109 + 5.45216 * Sppt + 1.4409e-2 * Sppt ** 2) / (10004.75 + 182.283 * Sppt + Sppt ** 2)

    # Eq. 18
    alpha0 = (6.9431 + 3.2841 * Sppt - 9.9486e-2 * Sppt ** 2) / (84.850 + 69.024 * Sppt + Sppt ** 2)
    alpha1 = 49.843 - 0.2276 * Sppt + 0.198e-2 * Sppt ** 2

    # Eq. 17
    RtR15 = 1.0 + (tempC - 15.0) * alpha0 / (alpha1 + tempC)

    # Eq. 14
    sigma = sigma35 * R15 * RtR15

    # Eq. 20
    a = 1.0 - Sppt * (3.838e-2 + 2.180e-3 * Sppt) * (79.88 + tempC) / ((12.01 + Sppt) * (52.53 + tempC))

    # Eq. 21
    b1 = (3.409e-2 + 2.817e-3 * Sppt) / (7.690 + Sppt)
    b2 = tempC * (2.46e-3 + 1.41e-3 * tempC) / (188.0 - 7.57 * tempC + tempC ** 2)
    b = 1.0 - Sppt * (b1 - b2)

    # Eq. 3
    eps_s = eps_s0 * a
    tau1x2pi = tau10x2pi * b

    # Eq. 22
    eps1 = 7.87e-2 * eps_s

    # Eq. 2
    term1 = (eps_s - eps1) / (1.0 - 1j * tau1x2pi * freqGHz)
    term2 = (eps1 - eps_inf) / (1.0 - 1j * tau2x2pi * freqGHz)
    term3 = 1j * sigma * 17.97510 / freqGHz

    return eps_inf + term1 + term2 + term3
