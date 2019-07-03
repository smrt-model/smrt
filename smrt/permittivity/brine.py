import warnings

import numpy as np

from smrt.core.error import SMRTError
from smrt.core.globalconstants import DENSITY_OF_ICE, FREEZING_POINT, PSU
from ..core.layer import layer_properties


def brine_conductivity(temperature):
    """computes ionic conductivity of dissolved salts, Stogryn and Desargant, 1985

    :param temperature: thermometric temperature [K]"""

    tempC = temperature - FREEZING_POINT  # temperature in deg Celsius
    if tempC >= -22.9:
        sigma = -tempC * np.exp(0.5193 + 0.08755 * tempC)
    else: # tempC < -22.9
        sigma = -tempC * np.exp(1.0334 + 0.1100 * tempC)
    return sigma


def brine_relaxation_time(temperature):
    """computes relaxation time of brine, Stogryn and Desargant, 1985

    :param temperature: thermometric temperature [K]"""

    tempC = temperature - FREEZING_POINT  # temperature in deg Celsius

    # Eq. (12) in Stogryn and Desargant given in seconds)
    tau_brine = 0.1099 + 0.13603e-2 * tempC + 0.20894e-3 * tempC ** 2 + 0.28167e-5 * tempC ** 3
    return tau_brine


@layer_properties("temperature")
def static_brine_permittivity(temperature):
    """computes  static dielectric constant of brine, Stogryn and Desargant, 1985

    :param temperature: thermometric temperature [K]"""

    tempC = temperature - FREEZING_POINT  # temperature in deg Celsius
    eps_static = (939.66 - 19.068 * tempC) / (10.737 - tempC)  # Static dielectric constant of saline water
    return eps_static


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


@layer_properties("temperature")
def permittivity_high_frequency_limit(temperature):
    """computes permittivity.

        :param temperature: ice or snow temperature in K"""

    tempC = temperature - FREEZING_POINT  # temperature in deg Celsius
    eps_inf = (82.79 + 8.19 * tempC ** 2) / (15.68 + tempC ** 2)
    return eps_inf


def brine_volume(temperature, salinity, porosity=0, bulk_density=None):
    """computes brine volume fraction using coefficients from Cox and Weeks (1983): 'Equations for determining the gas and brine volumes in sea-ice samples',
    J. of Glac. if ice temperature is below -2 deg C or coefficients determined by Lepparanta and Manninen (1988):
    'The brine and gas content of sea ice with attention to low salinities and high temperatures' for warmer temperatures.

    :param temperature: ice temperature in K
    :param salinity: salinity of ice in kg/kg (see PSU constant in smrt module)
    :param porosity: fractional air volume in ice (0..1). Default is 0.
    :param bulk_density: density of bulk ice in kg m :sup:`-3`
    """

    if temperature > calculate_freezing_temperature(salinity):
        return 1.  # if temperature of ice is above freezing temperature, which
    # is determined by salinity of the ice, brine volume fraction is set to 1, meaning that the saline ice is liquid (= saline water)

    T = temperature - FREEZING_POINT  # ice temperature in deg Celsius

    if T < -30.:
        warnings.warn("Temperature is below -30 deg C. Equation for calculating brine volume fraction is stated to be valid for temperatures from -30 to -2 deg C!")

    if T < -38.:
        raise SMRTError("(Polynomial) equations by Cox and Weeks (1983) were developed for temperatures between -30 and -2 deg C and show unphysical behaviour for temperatures lower than -38 deg C!")

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

    if bulk_density is None:
        bulk_density = (1 - porosity) * rho_ice * F1 / (
                    F1 - rho_ice * salinity * PSU ** -1 * F2) * 1e3  # bulk density of sea ice in kg/m3 (Cox and Weeks, 1983)
    elif porosity > 0:
        raise SMRTError("Calling brine_volume with both arguments bulk_density and porosity is ambigous. One is deduced from the other one.")

    Vb = salinity / PSU * bulk_density * 1e-3 / F1  # brine volume fraction (Cox and Weeks, 1983)

    if Vb > 1. and abs(temperature - calculate_freezing_temperature(salinity)) < 0.1:
        Vb = 1.  # the polynomial equations for
        # calculating brine volume fraction reach and exceed values of 1 slightly for temperatures slightly lower than the
        # calculated freezing temperature. If we are close to the freezing point (difference < 0.1K), we just set brine volume
        # fraction manually to 1.

    if Vb < 0 or Vb > 1:  # unphysical behaviour of polynomial equations (see error below):
        raise SMRTError("(Polynomial) equations for calculating brine volume fraction from temperature and salinity show \
        unphysical behaviour! Calculated value for brine volume fraction is below 0 or above 1!")

    return Vb


@layer_properties("salinity")
def calculate_freezing_temperature(salinity):
    """calculates temperature at which saline water freezes using polynomial fits
    of the Gibbs function given in TEOS-10: The international thermodynamic equation
    of seawater - 2010 ('http://www.teos-10.org/pubs/TEOS-10_Manual.pdf).
    The error of this fit ranges between -5e-4 K and 6e-4 K when compared with the
    temperature calculated from the exact in-situ freezing temperature, which is found
    by a Newton-Raphson iteration of the equality of the chemical potentials of water
    in seawater and in ice.

    :param salinity: salinity of ice in kg/kg (see PSU constant in smrt module)"""

    #Coefficients for polynomial:
    c0 =  0.017947064327968736
    c1 =  -6.076099099929818
    c2 =   4.883198653547851
    c3 =  -11.88081601230542
    c4 =   13.34658511480257
    c5 =  -8.722761043208607
    c6 =   2.082038908808201
    c7 =  -7.389420998107497
    c8 =  -2.110913185058476
    c9 =   0.2295491578006229
    c10 = -0.9891538123307282
    c11 = -0.08987150128406496
    c12 =  0.3831132432071728
    c13 =  1.054318231187074
    c14 =  1.065556599652796
    c15 = -0.7997496801694032
    c16 =  0.3850133554097069
    c17 = -2.078616693017569
    c18 =  0.8756340772729538
    c19 = -2.079022768390933
    c20 =  1.596435439942262
    c21 =  0.1338002171109174
    c22 =  1.242891021876471

    p = 10.1325 #pressure at sea level in dbar
    s_r	= salinity * 1e1 
    x = np.sqrt(s_r)
    p_r	= p * 1e-4

    #freezing temperature in deg Celsius:
    T_freeze = c0 + s_r*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + c6*x)))))+\
      p_r*(c7 + p_r*(c8 + c9*p_r)) + \
      s_r*p_r*(c10 + p_r*(c12+ p_r*(c15 + c21*s_r))+ s_r*(c13 + c17*p_r + c19*s_r)+\
                x*(c11 + p_r*(c14 + c18*p_r) + s_r*(c16 + c20*p_r + c22*s_r)))

    #return freezing temperature in K:
    return T_freeze + 273.15 
