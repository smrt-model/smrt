import numpy as np
import scipy as sc
import warnings
from ..core.layer import required_layer_properties
from smrt.core.globalconstants import FREEZING_POINT
from smrt.permittivity.ice import ice_permittivity_maetzler06

def brine_conductivity(temperature):
    """computes ionic conductivity of dissolved salts, Stogryn and Desargant, 1985 
    :param temperature: thermometric temperature [K]"""

    T = temperature - FREEZING_POINT  # temperature in deg Celsius
    if T >= -22.9:
        sigma = -T * np.exp(0.5193 + 0.08755 * T)
    if T < -22.9:
        sigma = -T * np.exp(1.0334 + 0.1100 * T)
    return sigma


def relaxation_time(temperature):
    """computes relaxation time of brine, Stogryn and Desargant, 1985
    :param temperature: thermometric temperature [K]"""

    T = temperature - FREEZING_POINT  # temperature in deg Celsius
    tau = (0.10990 + (0.13603E-2) * T + (0.20894E-3)
           * (T**2) + (0.28167E-5) * (T**3)) * 10**(-9)  # = 2 * pi * tau (Eq. (12) in Stogryn and Desargant given in nanoseconds)
    return tau


def static_brine_permittivity(temperature):
    """computes  static dielectric constant of brine, Stogryn and Desargant, 1985 
    :param temperature: thermometric temperature [K]"""

    T = temperature - FREEZING_POINT  # temperature in deg Celsius
    eps_static = (939.66 - 19.068 * T) / (10.737 - T)
    return eps_static


@required_layer_properties("temperature")
def brine_permittivity_stogryn85(frequency, temperature):
    """computes permittivity and loss of brine using equations given in Stogryn and Desargant (1985): 'The Dielectric Properties of Brine in Sea Ice at Microwave Frequencies', IEEE.
    :param frequency: em frequency [Hz]
    :param temperature: ice temperature in K"""

    eps_static = static_brine_permittivity(temperature)  # limiting static permittivity
    tau = relaxation_time(temperature)  # relaxation time
    sigma = brine_conductivity(temperature)  # ionic conductivity of dissolved salts

    T = temperature - FREEZING_POINT  # temperature in deg Celsius
    eps_inf = (82.79 + 8.19 * T**2) / (15.68 + T**2)  # limiting high frequency value
    e0 = 8.854e-12  # permittivity of free space

    brine_permittivity = eps_inf + (eps_static - eps_inf) / (1. - tau * frequency *
                                                             1j) + sigma / (2. * sc.pi * e0 * frequency) * 1j
    return brine_permittivity


def brine_volume(temperature, salinity):
    """computes brine volume fraction using coefficients from Cox and Weeks (1983): 'Equations for determining the gas and brine volumes in sea-ice samples', J. of Glac. if ice temperature is below -2 deg C or coefficients determined by Lepparanta and Manninen (1988): 'The brine and gas content of sea ice with attention to low salinities and high temperatures' for warmer temperatures.
    :param temperature: ice temperature in K
    :param salinity: salinity of ice in kg/kg (see PSU constant in smrt module)
    """

    T = temperature - FREEZING_POINT  # ice temperature in deg Celsius

    rho_ice = 0.917 - 1.403e-4 * T  # density of pure ice from Pounder, 1965

    if T < -2.:  # coefficients from Cox and Weeks, 1983
        if T >= -22.9:
            a0 = -4.732
            a1 = -2.245e1
            a2 = -6.397e-1
            a3 = -1.074e-2
            b0 = 8.903e-2
            b1 = -1.763e-2
            b2 = -5.33e-4
            b3 = -8.801e-6
        else:
            a0 = 9.899e3
            a1 = 1.309e3
            a2 = 5.527e1
            a3 = 7.160e-1
            b0 = 8.547
            b1 = 1.089
            b2 = 4.518e-2
            b3 = 5.819e-4
            
    elif T >= -2.:  # coefficients from Lepparanta and Manninen, 1988 for warm, low-salinity sea ice (e.g. Baltic sea ice)
        if T > 0:
            print("Warning! Ice temperature is above O deg C!")
        a0 = -4.1221e-2
        a1 = -1.8407e1
        a2 = 5.8402e-1
        a3 = 2.1454e-1
        b0 = 9.0312e-2
        b1 = -1.6111e-2
        b2 = 1.2291e-4
        b3 = 1.3603e-4

    F1 = np.polyval([a3, a2, a1, a0], T)
    F2 = np.polyval([b3, b2, b1, b0], T)
    rho_bulk = rho_ice * F1 / (F1 - rho_ice * salinity * 1e3 * F2)  # bulk density of sea ice (Cox and Weeks, 1983)
    Vb = salinity * 1e3 * rho_bulk / F1  # brine volume fraction (Cox and Weeks, 1983)

    # Polynoms can give values >1 or <0 for high temperatures approaching (or exceeding) 0 deg C -->
    if Vb > 1.:
        Vb = 1.  # brine volume fraction cannot be higher than 1
    elif Vb < 0:
        Vb = 0.  # brine volume fraction cannot be lower than 0

    return Vb


@required_layer_properties("temperature", "salinity")
def impure_ice_permittivity_maetzler06(frequency, temperature, salinity):
    """ Computes permittivity of impure ice from Maetzler 2006 - Thermal Microwave Radiation: Applications for Remote Sensing

        Model developed for salinity around 0.013 PSU, so it is not recommended for sea ice
        :param temperature: ice temperature in K
        :param salinity: salinity of ice in kg/kg (see PSU constant in smrt module)

    **Usage example**::

        from smrt.permittivity.saline_ice import impure_ice_permittivity_maetzler06
        eps_ice = impure_ice_permittivity_maetzler06(frequency=18e9, temperature=270, salinity=0.013)

    """

    # Issue warning if salinity > 0.013 PSU
    if salinity > 0.013e-3:
        warnings.warn("This permittity model was developed for saline impurities of around 0.013 10^-3 kg/kg (or 0.013 PSU)")

    # Modify imaginary component calculated for pure ice 
    pure_ice_permittivity = ice_permittivity_maetzler06(frequency, temperature)
    freqGHz = frequency * 10e-9
    # Equation 5.37 from Maetzler 2006: Thermal Microwave Radiation: Applications for Remote Sensing
    g0 = 1866 * np.exp(-0.317 * freqGHz)
    g1 = 72.2 + 6.02 * freqGHz
    # Equation 5.36
    delta_Eimag = 1. / (g0 + g1 * (FREEZING_POINT - temperature))
    # Equation 5.38
    S0 = 0.013 # CURRENT UNITS ARE PSU
    #print (pure_ice_permittivity, delta_Eimag * salinity * 1e3 / S0)
    #print (pure_ice_permittivity + 1j * delta_Eimag * salinity * 1e3 / S0)
    return pure_ice_permittivity + 1j * delta_Eimag * salinity * 1e3 / S0
