
import numpy as np
from ..core.layer import required_layer_properties
from ..core.error import SMRTError
from .saline_water import seawater_permittivity_stogryn71, seawater_permittivity_stogryn95


@required_layer_properties("density", "temperature", "salinity")
def saline_snow_permittivity_geldsetzer09(frequency, density, temperature, salinity):
    """ Computes permittivity of saline snow using the frequency dispersion model published by Geldsetzer et al., 2009 (CRST). DOI: 10.1016/j.coldregions.2009.03.009.
    In-situ measurements collected had salinity concentration between 0.1e-3 and 12e3 kg/kg, temperatures ranging between 257 and 273 K, and a mean snow density of 352 kg/m3.

    Validity between 10 MHz and 40 GHz.

    Converted from matlab code created in 08.2011 by Ludovic Brucker, NASA GSFC code 615

        :param temperature: ice temperature in K
        :param salinity: salinity of ice in kg/kg (see PSU constant in smrt module)

    """

    # %         Sppt= snow salinity in ppt
    freqGHz = frequency * 1e-9
    Sppt = salinity * 1e3
    tempC = temperature - 273.15

    eps_drysnow = 1 + 2.55 * (density / 1e3)
    einf = (82.79 + 8.19 * tempC**2) / (15.68 + tempC**2)
    es = (939.66 - 19.068 * tempC) / (10.737 - tempC)
    omega_b = 2 * np.pi * frequency

    tau_b = 0.1099 + 0.13603e-2 * tempC + 0.20894e-3 * tempC**2 + 0.28167e-5 * tempC**3
    fr = 1 / tau_b
    sigma_b = -tempC * np.exp(0.5193 + 0.08755 * tempC)

    if tempC > -8 and tempC <= -2:
        Sb = 1.725 - 18.756 * tempC - 0.3946 * tempC**2
    else:
        Sb = 57.041 - 9.929 * tempC - 0.16204 * tempC**2 - 0.002396 * tempC**3

    vb = 1e-3 * Sppt * (-49.185 / tempC + 0.532)
    rho_i = 917 - 0.1403 * tempC
    rho_b = 1e3 + 0.8 * Sb
    Vb = (vb * rho_b) / ((1 - vb) * rho_i + vb * rho_b) * (density / rho_b)

    real_b = einf + (es-einf) / (1 + (freqGHz / fr)**2)
    real_mix = eps_drysnow + 1.33 * Vb * real_b
    lossb_rel = (es-einf) * (freqGHz / fr) / (1 + (freqGHz / fr)**2)
    lossb_con = sigma_b / (omega_b * 0.00000000000885419)
    lossmix_con = lossb_con * Vb**1.778
    loss_mix = 0.002 + 1.33 * Vb * lossb_rel+lossmix_con

    return real_mix + 1j* loss_mix



@required_layer_properties("density", "temperature", "salinity")
def saline_snow_permittivity_scharien_with_stogryn71(frequency, density, temperature, salinity):
    """Computes permittivity of saline snow. See `saline_snow_permittivity_scharien` documentation"""

    return saline_snow_permittivity_scharien(frequency, density, temperature, salinity,
                                             seawater_permittivity_stogryn71(frequency, temperature))


@required_layer_properties("density", "temperature", "salinity")
def saline_snow_permittivity_scharien_with_stogryn95(frequency, density, temperature, salinity):
    """Computes permittivity of saline snow. See `saline_snow_permittivity_scharien` documentation"""

    return saline_snow_permittivity_scharien(frequency, density, temperature, salinity,
                                             seawater_permittivity_stogryn95(frequency, temperature, salinity))



def saline_snow_permittivity_scharien(frequency, density, temperature, salinity, brine_permittivity):

#####    (f, Ts, rho_s, Ss, opt, epsiMEMLS, seawater)

    """Computes permittivity of saline snow using the Denoth / Matzer Mixture Model - Dielectric Contsant of Saline Snow
    
     R.Scharien, CEOS, University of Manitoba
     scharien @ cc.umanitoba.ca
     Last update: July 05, 2011
    
     Assumptions:
     (1) Brine inclusion geometry as oblate spheroids
         Depolarization factor, A0 = 0.053(Denoth, 1980)
     (2) Brine inclusions are isotropically oriented
         Coupling factor, X = 2 / 3(Drinkwater and Crocker, 1988)

     Validity ranges:
    (1) Temperature, Ts, down to - 22.9 degrees Celcius;
    (2) Brine salinity, Sb, up to 157ppt;  i.e.up to a Normality of 3 for NaCl
    % % % % % Not valid for wet snow % % % %
        %
 % Input parameters:
%       Snow sample temperature, Ts, in kelvin
%       Snow sample density, rho_s, in kg m-3
%       Snow sample salinity, Ss, in ppt.
%       Frequency at which to evaluate, f, in GHz
%%
"""
    # convert units
    tempC = temperature - 273.15

    Ss = salinity * 1e3

    # True brine volume of snow(Vb)
    # density of pure ice, rho_i
    rho_i = 917 - 0.1403 * tempC

    # density of brine in snow, rho_b
    # requires brine salinity, Sb, in ppt.(Cox and Weeks, 1975)


    if tempC > -2:
        Sb = 0.02515 - 17.787 * tempC
    elif tempC >= -8.2:
        Sb = 1.725 - 18.756 * tempC - 0.3946 * tempC**2
    else:
        Sb = 57.041 - 9.929 * tempC - 0.16204 * tempC**2 - 0.002396 * tempC**3

    rho_b = 1000 + 0.8 * Sb  # density of brine

    # initial brine volume of salt / ice mixture, vb
    if tempC >= -0.1:
        vb = Ss * 500.9
    elif tempC >= -0.2:
        vb = Ss * 250.5
    elif tempC >= -0.3:
        vb = Ss * 167.1
    elif tempC >= -0.4:
        vb = Ss * 125.4
    else:
        vb = Ss * (-49.185 / tempC + 0.532)

    vb *= 1e-3

    if (tempC < -22.9) and (salinity ==0):
        raise SMRTError('Snow temperatures are too low to be valid for calculating correct brine density and volume in snow.')

    # true brine volume fraction, Vb
    Vb = ((vb * rho_b) / ((1 - vb) * rho_i + vb * rho_b)) * (density / rho_b)

    # Mixture model for dielectric constant of saline snow
    #if opt == 1:
    rho_drysnow = density - Vb * rho_b  # Equivalent dry snow density; removal of brine from original snow measurement

    # Permittivity of dry snow; density dependent

    if rho_drysnow <= 500:
        eps_drysnow = 1 + 1.9 * (rho_drysnow / 1000)
    else:
        eps_drysnow = 0.51 + 2.88 * (rho_drysnow / 1000)
    #else:  ## don't implement this option
    #    eps_drysnow = epsiMEMLS;

    A0 = 0.053  # Depolarization factor
    X = 0.667  # Coupling factor

    # Output dielectric constant of mixture

    epsBRINE = eps_drysnow + ((X * Vb) * ((brine_permittivity - eps_drysnow) / (1 + (brine_permittivity / eps_drysnow - 1) * A0)))

    return epsBRINE


