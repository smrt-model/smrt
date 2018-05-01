import warnings

import numpy as np

from smrt.core.error import SMRTError
from smrt.core.globalconstants import DENSITY_OF_ICE, FREEZING_POINT
from ..core.layer import required_layer_properties


@required_layer_properties("temperature")
def brine_conductivity(temperature):
    """computes ionic conductivity of dissolved salts, Stogryn and Desargant, 1985
    :param temperature: thermometric temperature [K]"""

    tempC = temperature - FREEZING_POINT  # temperature in deg Celsius
    if tempC >= -22.9:
        sigma = -tempC * np.exp(0.5193 + 0.08755 * tempC)
    else: # tempC < -22.9
        sigma = -tempC * np.exp(1.0334 + 0.1100 * tempC)
    return sigma


@required_layer_properties("temperature")
def brine_relaxation_time(temperature):
    """computes relaxation time of brine, Stogryn and Desargant, 1985
    :param temperature: thermometric temperature [K]"""

    tempC = temperature - FREEZING_POINT  # temperature in deg Celsius

    # Eq. (12) in Stogryn and Desargant given in seconds)
    tau_brine = 0.1099 + 0.13603e-2 * tempC + 0.20894e-3 * tempC ** 2 + 0.28167e-5 * tempC ** 3
    return tau_brine


@required_layer_properties("temperature")
def static_brine_permittivity(temperature):
    """computes  static dielectric constant of brine, Stogryn and Desargant, 1985
    :param temperature: thermometric temperature [K]"""

    tempC = temperature - FREEZING_POINT  # temperature in deg Celsius
    eps_static = (939.66 - 19.068 * tempC) / (10.737 - tempC)  # Static dielectric constant of saline water
    return eps_static


@required_layer_properties("temperature")
def calculate_brine_salinity(temperature):
    """ Computes the salinity of brine (in ppt) for a given temperature (Cox and Weeks, 1975)
        :param temperature: snow temperature in K
        :return salinity_brine in ppt
    """
    tempC = temperature - FREEZING_POINT

    if tempC > -2:
        salinity_brine = 0.02515 - 17.787 * tempC
    elif tempC >= -8.2:
        salinity_brine = 1.725 - 18.756 * tempC - 0.3946 * tempC ** 2
    else:
        salinity_brine = 57.041 - 9.929 * tempC - 0.16204 * tempC ** 2 - 0.002396 * tempC ** 3

    return salinity_brine


@required_layer_properties("temperature")
def permittivity_high_frequency_limit(temperature):
    """computes permittivity .
        :param temperature: ice or snow temperature in K"""

    tempC = temperature - FREEZING_POINT  # temperature in deg Celsius
    eps_inf = (82.79 + 8.19 * tempC ** 2) / (15.68 + tempC ** 2)
    return eps_inf


@required_layer_properties("temperature", "salinity")
def brine_volume(temperature, salinity):
    """computes brine volume fraction using coefficients from Cox and Weeks (1983): 'Equations for determining the gas and brine volumes in sea-ice samples', J. of Glac. if ice temperature is below -2 deg C or coefficients determined by Lepparanta and Manninen (1988): 'The brine and gas content of sea ice with attention to low salinities and high temperatures' for warmer temperatures.
    :param temperature: ice temperature in K
    :param salinity: salinity of ice in kg/kg (see PSU constant in smrt module)"""

    T = temperature - FREEZING_POINT  # ice temperature in deg Celsius

    if T < -30.:
        warnings.warn(
                "Temperature is below -30 deg C. Equation for calculating brine volume fraction is stated to be valid for temperatures from -30 to -2 deg C!")

    if T < -38.:
        raise SMRTError(
                "(Polynomial) equations by Cox and Weeks (1983) were developed for temperatures between -30 and -2 deg C and show unphysical behaviour for temperatures lower than -38 deg C!")

    rho_ice = DENSITY_OF_ICE / 1e3 - 1.403e-4 * T  # density of pure ice from Pounder, 1965

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

    if T > -2.15 and (
            Vb > 1. or Vb < 0.):  # limitations for physical behaviour of polynomial equations (see warning below):
        Vb = 1
        warnings.warn(
                "(Polynomial) equations for calculating brine volume fraction from temperature and salinity show unphysical behaviour for high temperatures approaching (or exceeding) 0 deg C. If temperature reaches values > freezing temperature (which is a function of salinity), the ice melts and only liquid water = brine is left. This happened here and brine volume fraction was set to 1 manually.")

    return Vb
