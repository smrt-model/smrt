import warnings

import numpy as np

from smrt.core.globalconstants import DENSITY_OF_ICE, FREEZING_POINT, GHz, PERMITTIVITY_OF_FREE_SPACE, PSU
from .brine import brine_conductivity, brine_relaxation_time, calculate_brine_salinity, \
    permittivity_high_frequency_limit, static_brine_permittivity
from .saline_water import seawater_permittivity_stogryn71, seawater_permittivity_stogryn95
from ..core.error import SMRTError
from ..core.layer import layer_properties


@layer_properties("density", "temperature", "salinity")
def saline_snow_permittivity_geldsetzer09(frequency, density, temperature, salinity):
    """ Computes permittivity of saline snow using the frequency dispersion model published by Geldsetzer et al., 2009 (CRST). DOI: 10.1016/j.coldregions.2009.03.009.
    In-situ measurements collected had salinity concentration between 0.1e-3 and 12e3 kg/kg, temperatures ranging between 257 and 273 K, and a mean snow density of 352 kg/m3.

    Validity between 10 MHz and 40 GHz.

    Source: Matlab code, Ludovic Brucker

    :param frequency: frequency in Hz
    :param density: snow density in kg m-3
    :param temperature: ice temperature in K
    :param salinity: salinity of ice in kg/kg (see PSU constant in smrt module)
    """

    if np.max(frequency) > 40e9:
        warnings.warn(
                "The permittivity model of Geldsetzer et al. 2009 "
                "(doi:10.1016/j.coldregions.2009.03.009) "
                "was developed for the frequency range 10 MHz - 40 GHz.")

    freqGHz = frequency / GHz
    tempC = temperature - FREEZING_POINT

    eps_drysnow = 1 + 2.55 * (density / 1e3)
    eps_inf = permittivity_high_frequency_limit(temperature)
    eps_static = static_brine_permittivity(temperature)
    omega_brine = 2 * np.pi * frequency
    tau_brine = brine_relaxation_time(temperature)
    fr = 1 / tau_brine
    sigma_brine = brine_conductivity(temperature)
    salinity_brine = calculate_brine_salinity(temperature)

    # Initial brine volume
    initial_brine_volume = salinity * (-49.185 / tempC + 0.532)
    density_ice = DENSITY_OF_ICE - 0.1403 * tempC
    density_brine = 1e3 + 0.8 * salinity_brine

    # True brine volume fraction, Vb
    true_brine_volume = (initial_brine_volume * density_brine) / (
            (1 - initial_brine_volume) * density_ice + initial_brine_volume * density_brine) * (
                                density / density_brine)

    real_brine = eps_inf + (eps_static - eps_inf) / (1 + (freqGHz / fr) ** 2)
    real_mix = eps_drysnow + 1.33 * true_brine_volume * real_brine
    lossb_rel = (eps_static - eps_inf) * (freqGHz / fr) / (1 + (freqGHz / fr) ** 2)
    lossb_con = sigma_brine / (omega_brine * PERMITTIVITY_OF_FREE_SPACE)
    lossmix_con = lossb_con * true_brine_volume ** 1.778
    loss_mix = 0.002 + 1.33 * true_brine_volume * lossb_rel + lossmix_con

    return real_mix + 1j * loss_mix


@layer_properties("density", "temperature", "salinity")
def saline_snow_permittivity_scharien_with_stogryn71(frequency, density, temperature, salinity):
    """Computes permittivity of saline snow. See `saline_snow_permittivity_scharien` documentation"""

    return saline_snow_permittivity_scharien(density, temperature, salinity,
                                             seawater_permittivity_stogryn71(frequency, temperature))


@layer_properties("density", "temperature", "salinity")
def saline_snow_permittivity_scharien_with_stogryn95(frequency, density, temperature, salinity):
    """Computes permittivity of saline snow. See `saline_snow_permittivity_scharien` documentation"""

    return saline_snow_permittivity_scharien(density, temperature, salinity,
                                             seawater_permittivity_stogryn95(frequency, temperature, salinity))


def saline_snow_permittivity_scharien(density, temperature, salinity, brine_permittivity):
    """Computes permittivity of saline snow using the Denoth / Matzler Mixture Model - Dielectric Contsant of Saline Snow.

     Assumptions:
     (1) Brine inclusion geometry as oblate spheroids
         Depolarization factor, A0 = 0.053 (Denoth, 1980)
     (2) Brine inclusions are isotropically oriented
         Coupling factor, X = 2/3 (Drinkwater and Crocker, 1988)

     Validity ranges:
    (1) Temperature, Ts, down to - 22.9 degrees Celcius;
    (2) Brine salinity, Sb, up to 157ppt;  i.e.up to a Normality of 3 for NaCl
    Not valid for wet snow

    Source: Matlab code, Randall Scharien

    :param density: snow density in kg m-3
    :param temperature: snow temperature in K
    :param salinity: snow salinity in kg/kg (see PSU constant in smrt module)
    :param brine_permittivity: brine_permittivity
    """

    # convert units
    tempC = temperature - FREEZING_POINT
    Sppt = salinity / PSU  # snow salinity in ppt

    # True brine volume of snow(Vb)
    # density of pure ice, density_ice
    density_ice = DENSITY_OF_ICE - 0.1403 * tempC

    # density of brine in snow requires brine salinity in ppt.(Cox and Weeks, 1975)
    salinity_brine = calculate_brine_salinity(temperature)
    density_brine = 1000 + 0.8 * salinity_brine

    # initial brine volume
    if tempC >= -0.1:
        initial_brine_volume = Sppt * 500.9
    elif tempC >= -0.2:
        initial_brine_volume = Sppt * 250.5
    elif tempC >= -0.3:
        initial_brine_volume = Sppt * 167.1
    elif tempC >= -0.4:
        initial_brine_volume = Sppt * 125.4
    else:
        initial_brine_volume = Sppt * (-49.185 / tempC + 0.532)

    initial_brine_volume *= PSU

    if (tempC < -22.9) and (salinity == 0):
        raise SMRTError(
                'Snow temperatures may be too low to be valid for calculating correct brine density and volume in snow.')

    # true brine volume fraction, initial_brine_volume
    true_brine_volume = ((initial_brine_volume * density_brine) / (
            (1 - initial_brine_volume) * density_ice + initial_brine_volume * density_brine)) * (
                                density / density_brine)

    # Equivalent dry snow density; removal of brine from original snow density
    density_drysnow = density - true_brine_volume * density_brine

    # Permittivity of dry snow; density dependent
    if density_drysnow <= 500:
        eps_drysnow = 1 + 1.9 * (density_drysnow / 1000)
    else:
        eps_drysnow = 0.51 + 2.88 * (density_drysnow / 1000)

    depolarization_factor = 0.053
    coupling_factor = 0.667

    # Output dielectric constant of mixture
    eps_brine = eps_drysnow + (
            (coupling_factor * true_brine_volume) * ((brine_permittivity - eps_drysnow) / (
            1 + (brine_permittivity / eps_drysnow - 1) * depolarization_factor)))

    return eps_brine
