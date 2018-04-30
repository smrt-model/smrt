# coding: utf-8

import numpy as np

from ..core.layer import required_layer_properties


@required_layer_properties("temperature", "salinity")
def seawater_permittivity_klein76(frequency, temperature, salinity):
    """Calculates permittivity (dielectric constant) of water using an empirical relationship described
       by Klein and Swift (1976).
       :param frequency: frequency in Hz
       :param temperature: water temperature in K
       :param salinity: water salinity in kg/kg (see PSU constant in smrt module)
       Returns complex water permittivity for a frequency f."""

    T = temperature - 273.15
    S = salinity * 1e3
    f = frequency

    omega = 2 * np.pi * f
    eps_0 = 8.854e-12  # permittivity of free space
    eps_inf = 4.9  # limiting high frequency value

    # calculate eps_s = static dielectric constant of saline water:
    eps_s_T = 87.134 - 1.949e-1 * T - 1.276e-2 * T**2 + 2.491e-4 * T**3
    a_ST = 1. + 1.613e-5 * S * T - 3.656e-3 * S + 3.210e-5 * S**2 - 4.232e-7 * S**3
    eps_s = eps_s_T * a_ST

    # calculate tau = relaxation time of saline water:
    tau_T0 = 1.768e-11 - 6.086e-13 * T + 1.104e-14 * T**2 - 8.111e-17 * T**3
    b_ST = 1. + 2.282e-5 * S * T - 7.638e-4 * S - 7.760e-6 * S**2 + 1.105e-8 * S**3
    tau = tau_T0 * b_ST

    # calculate sigma = ionic conductivity of dissolved salts:
    delta = 25 - T
    beta = 2.0333e-2 + 1.266e-4 * delta + 2.464e-6 * delta**2 - S * \
      (1.849e-5 - 2.551e-7 * delta + 2.551e-8 * delta**2)
    sigma_25S = S * (0.182521 - 1.46192e-3 * S + 2.09324e-5 * S**2 - 1.28205e-7 * S**3)
    sigma = sigma_25S * np.exp(-delta * beta)

    # Debye type relaxation equation:
    eps_water = eps_inf + (eps_s - eps_inf) / (1 - 1j *
                                               omega * tau) + 1j * sigma / (omega * eps_0)
    return eps_water


@required_layer_properties("temperature")
def seawater_permittivity_stogryn71(frequency, temperature):
    """Compute dielectric constant of brine, complex_b (Stogryn, 1971 approach)

    Input parameters: from polynomial fiT in Stogryn and Desargent, 1985
    
    created on: 09.2011 by Ludo Brucker
    converted from matlab by Ghislain Picard
    
    :param frequency: frequency in Hz
    :param temperature: water temperature in K

"""
    f = frequency * 1e-9
    tempC = temperature - 273.15

    einf = (82.79 + 8.19 * tempC**2) / (15.68 + tempC**2)  # High-frequency dielectric constant of saline water
    es = (939.66 - 19.068 * tempC) / (10.737 - tempC)  # Static dielectric constant of saline water
    e0 = 0.00000000000885419  # Permittivity of free space
    omega_b = 2 * np.pi * frequency  # Angular frequency
    tau_b = 0.1099 + 0.13603e-2 * tempC + 0.20894e-3 * tempC**2 + 0.28167e-5 * tempC**3  # Relaxation time;
    # Eq. 12 in Stogryn and Desargent, 1985
    sigma_b = -tempC * np.exp(0.5193 + 0.8755e-1 * tempC)  # Ionic conductivity (dissolved salT) to -22.9degC

    # Output dielectric constant of brine using Stogryn, 1971 formulation (Ulaby et al., 1986 p. 2046)
    real_b = einf + ((es - einf) / (1 + (tau_b * f)**2))
    imag_b = (tau_b * f) * ((es-einf) / (1+(tau_b * f)**2)) + (sigma_b / (omega_b * e0))

    return real_b + 1j * imag_b



@required_layer_properties("temperature", "salinity")
def seawater_permittivity_stogryn95(frequency, temperature, salinity):

    """compute seawater dielectric constant using Stogryn 1995.

    source: Stogryn 1995 + http: // rime.aos.wisc.edu / MW / models / src / eps_sea_stogryn.f90
    created on: 09.2011 by Ludo Brucker
    converted from matlab by Ghislain Picard

    :param frequency: frequency in Hz
    :param temperature: water temperature in K
    :param salinity: water salinity in kg/kg (see PSU constant in smrt module)

"""
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

#    f = frequency(GHz)
#    T = temperature(C)
#    S = salinity(ppt)

    f = frequency * 1e-9
    S = salinity * 1e3
    tempC = temperature - 273.15

    # Eq.6
    eps_s0 = (3.70886e4 - 8.2168e1 * tempC) / (4.21854e2 + tempC)

    # Eq. 8
    tau10x2pi = (255.04 + 0.7246 * tempC) / ((49.25 + tempC) * (45 + tempC))
    # Eq. 9
    tau2x2pi = 0.628e-2

    # Eq. 10
    eps_inf = 4.05 + 1.86e-2 * tempC

    # Eq. 15
    sigma35 = 2.903602 + 8.60700e-2 * tempC + 4.738817e-4 * tempC**2 - 2.9910e-6 * tempC**3 + 4.3047e-9 * tempC**4

    # Eq. 16
    R15 = S * (37.5109 + 5.45216 * S + 1.4409e-2 * S**2) / (10004.75 + 182.283 * S + S**2)

    # Eq. 18
    alpha0 = (6.9431 + 3.2841 * S - 9.9486e-2 * S**2) / (84.850 + 69.024 * S + S**2)
    alpha1 = 49.843 - 0.2276 * S + 0.198e-2 * S**2

    # Eq. 17
    RtR15 = 1.0 + (tempC - 15.0) * alpha0 / (alpha1 + tempC)

    # Eq. 14
    sigma = sigma35 * R15 * RtR15

    # Eq. 20
    a = 1.0 - S * (3.838e-2 + 2.180e-3 * S) * (79.88 + tempC) / ((12.01 + S) * (52.53 + tempC))

    # Eq. 21
    b1 = (3.409e-2 + 2.817e-3 * S) / (7.690 + S)
    b2 = tempC * (2.46e-3 + 1.41e-3 * tempC) / (188.0 - 7.57 * tempC + tempC**2)
    b = 1.0 - S * (b1 - b2)

    # Eq. 3
    eps_s = eps_s0 * a
    tau1x2pi = tau10x2pi * b

    # Eq. 22
    eps1 = 7.87e-2 * eps_s

    # Eq. 2
    term1 = (eps_s - eps1) / (1.0 - 1j * tau1x2pi * f)
    term2 = (eps1 - eps_inf) / (1.0 - 1j * tau2x2pi * f)
    term3 = 1j * sigma * 17.97510 / f
    
    return eps_inf + term1 + term2 + term3
