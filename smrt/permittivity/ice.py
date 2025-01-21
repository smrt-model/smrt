# coding: utf-8

from __future__ import print_function

# Stdlib import

# other import
import numpy as np

# local import
# from ..core.error import SMRTError
from ..core.globalconstants import FREEZING_POINT, DENSITY_OF_ICE
from ..core.layer import layer_properties
from ..core.error import SMRTError

#
# for developers: see note in __init__.py
#


@layer_properties("temperature")
def ice_permittivity_maetzler06(frequency, temperature):
    """ Calculates the complex ice dielectric constant depending on the frequency and temperature

    Based on Mätzler, C. (2006). Thermal Microwave Radiation: Applications for Remote Sensing p456-461
    This is the default model used in smrt.inputs.make_medium.make_snow_layer().

    :param frequency: frequency in Hz
    :param temperature: temperature in K
    :returns: complex permittivity of pure ice

    **Usage example**::

        from smrt.permittivity.ice import ice_permittivity_maetzler06
        eps_ice = ice_permittivity_maetzler06(frequency=18e9, temperature=270)

    .. note::

        Ice permittivity is automatically calculated in smrt.inputs.make_medium.make_snow_layer() and
        is not set by the electromagnetic model module. An alternative
        to ice_permittivity_maetzler06 may be specified as an argument to the make_snow_layer
        function. The usage example is provided for external reference or testing purposes.

"""

    freqGHz = frequency / 1e9

    tempC = temperature - FREEZING_POINT

    if np.any(tempC > 0):
        raise SMRTError(f"The ice temperature must be lower or equal to {FREEZING_POINT}K")

    Ereal = 3.1884 + 9.1e-4 * tempC

    theta = 300.0 / temperature - 1.0
    alpha = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)

    B1 = 0.0207
    B2 = 1.16e-11
    b = 335.
    deltabeta = np.exp(- 9.963 + 0.0372 * tempC)
    betam = (B1 / temperature) * (np.exp(b / temperature) / ((np.exp(b / temperature) - 1)**2)) + B2 * freqGHz**2
    beta = betam + deltabeta

    Eimag = alpha / freqGHz + beta * freqGHz

    return Ereal + 1j * Eimag


@layer_properties("temperature")
def ice_permittivity_maetzler98(frequency, temperature):
    """Computes permittivity of ice (accounting for ionic impurities in ice?), equations from Hufford (1991) as given in Maetzler (1998): 'Microwave properties of ice and snow', in B. Schmitt et al. (eds.): 'Solar system ices', p. 241-257, Kluwer.

    :param temperature: ice temperature in K
    :param frequency: Frequency in Hz"""

    tempC = temperature - FREEZING_POINT

    if np.any(tempC > 0):
        raise SMRTError(f"The ice temperature must be lower or equal to {FREEZING_POINT}K")

    f = frequency * 1e-9
    epi = 3.1884 + 9.1e-4 * tempC

    # The Hufford model for the imaginary part:
    theta = 300. / temperature - 1.
    alpha = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)
    beta = (0.502 - 0.131 * theta / (1 + theta)) * 1e-4 + \
        (0.542e-6 * ((1 + theta) / (theta + 0.0073))**2)

    epii = (alpha / f) + (beta * f)
    return epi + epii * 1j


@layer_properties("temperature")
def ice_permittivity_maetzler87(frequency, temperature):
    """Calculates the complex ice dielectric constant depending on the frequency and temperature

    Based on Mätzler, C. and Wegmüller (1987). Dielectric properties of fresh-water ice at microwave frequencies.
    J. Phys. D: Appl. Phys. 20 (1987) 1623-1630.

    :param frequency: frequency in Hz
    :param temperature: temperature in K
    :returns: complex permittivity of pure ice

    **Usage example**::

        from smrt.permittivity.ice import ice_permittivity_maetzler87
        eps_ice = ice_permittivity_maetzler87(frequency=18e9, temperature=270)

    .. note::

        This is only suitable for testing at -5 deg C and -15 deg C. If used at other temperatures
        a warning will be displayed.

"""

    import warnings

    freqGHz = frequency / 1e9

    tempC = temperature - FREEZING_POINT

    if np.any(tempC > 0):
        raise SMRTError(f"The ice temperature must be lower or equal to {FREEZING_POINT}K")

    # Equation 10
    Ereal = 3.1884 + 9.1e-4 * tempC

    if (temperature - FREEZING_POINT) < -10:
        A = 3.5e-4
        B = 3.6e-5
        C = 1.2
    else:
        A = 6e-4
        B = 6.5e-5
        C = 1.07
    # Equation 13
    Eimag = A / freqGHz + B * freqGHz**C
    # Issue warning if temperature different from values in paper
    if temperature not in [FREEZING_POINT-5, FREEZING_POINT-15]:
        warnings.warn("Strictly, this permittivity formulation was proposed for -5 and -15 deg C. It is recommended to use another formulation if this is not for testing purpose")

    return Ereal + Eimag * 1j


@layer_properties("temperature")
def ice_permittivity_tiuri84(frequency, temperature):
    """Calculates the complex ice dielectric constant depending on the frequency and temperature

    Based on Tiuri et al. (1984). The Complex Dielectric Constant of Snow at Microwave Frequencies.
    IEEE Journal of Oceanic Engineering, vol. 9, no. 5., pp. 377-382

    :param frequency: frequency in Hz
    :param temperature: temperature in K
    :returns: complex permittivity of pure ice

    **Usage example**::

        from smrt.permittivity.ice import ice_permittivity_tiuri84
        eps_ice = ice_permittivity_tiuri84(frequency=1.9e9, temperature=250)

"""

    tempC = temperature - FREEZING_POINT

    if np.any(tempC > 0):
        raise SMRTError(f"The ice temperature must be lower or equal to {FREEZING_POINT}K")

    # Units conversion
    density_gm3 = DENSITY_OF_ICE * 1e-3

    # Eq (1) - Real part
    Ereal = 1 + 1.7 * density_gm3 + 0.7 * density_gm3**2

    # Eq (6) - Imaginary part
    Eimag = 1.59e6 * \
        (0.52 * density_gm3 + 0.62*density_gm3**2) * \
        (frequency**-1 + 1.23e-14 * frequency**.5) * np.exp(0.036 * tempC)

    return Ereal + 1j * Eimag


@layer_properties("temperature")
def _ice_permittivity_HUT(frequency, temperature):
    # This gives exact agreement with the HUT model version
    # Only use if invoking an exact HUT simulation.
    # Should not be included in the sphinx documentation

    # Raise exception if temperature is zero
    # if temperature == 0:
    #    raise SMRTError('Temperature used in calculation of ice permittivity is zero')

    # Issue warning if temperature is below 240K (real part is for T > 240K)
    # if (temperature < 240).any:
    #    warnings.warn('Warning: temperature is below 240K. Ice permittivity is out of range of applicability')

    # Real part: from Mätzler and Wegmuller (1987)



    if np.any(temperature > 273):
        raise SMRTError(f"The ice temperature must be lower or equal to 273.0 K")

    real_permittivity_ice = 3.1884 + 9.1e-4 * (temperature - 273.0)

    # Imaginary part: from Mätzler (2006)
    # NB frequencies in original equations are in GHz, here in Hz.
    freqGHz = frequency * 1e-9
    theta = (300.0 / temperature) - 1.0  # Floats needed for correct calculation in py2.7 but not needed in 3.x
    alpha = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)
    beta = (0.0207 / temperature) * (np.exp(335.0 / temperature) / (np.exp(335.0 / temperature) - 1.0)**2) + (
        1.16e-11 * (freqGHz)**2 + np.exp(-10.02 + 0.0364 * (temperature - 273.0)))
    imag_permittivity_ice = alpha / freqGHz + beta * freqGHz

    return real_permittivity_ice + 1j * imag_permittivity_ice


@layer_properties("temperature")
def _ice_permittivity_DMRTML(frequency, temperature):
    # This gives exact agreement with the DMRT-ML model version
    # Only use if invoking an exact DMRT-ML simulation.
    # Should not be included in the sphinx documentation

    # Raise exception if temperature is zero
    # if temperature == 0:
    #     raise SMRTError('Temperature used in calculation of ice permittivity is zero')

    # Issue warning if temperature is below 240K (real part is for T > 240K)
    # if np.any(temperature < 240):
    #    warnings.warn('Warning: temperature is below 240K. Ice permittivity is out of range of applicability')

    # Real part: from Mätzler and Wegmuller (1987)

    real_permittivity_ice = 3.1884 + 9.1e-4 * (temperature - 273.0)
    # Imaginary part: from Mätzler (2006)
    # NB frequencies in original equations are in GHz, here in Hz.
    freqGHz = frequency * 1e-9
    theta = (300.0 / temperature) - 1.0  # Floats needed for correct calculation in py2.7 but not needed in 3.x
    alpha = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)
    beta = (0.0207 / temperature) * (np.exp(335.0 / temperature) / (np.exp(335.0 / temperature) - 1.0)**2) + (
        1.16e-11 * (freqGHz)**2 + np.exp(-9.963 + 0.0372 * (temperature - 273.16)))
    imag_permittivity_ice = alpha / freqGHz + beta * freqGHz

    return real_permittivity_ice + 1j * imag_permittivity_ice


@layer_properties("temperature", "salinity")
def _ice_permittivity_MEMLS(frequency, temperature, salinity):
    # This gives exact agreement with the MEMLS model version
    # Only use if invoking an exact MEMLS simulation.
    # Should not be included in the sphinx documentation
    # Note salinity should be in parts per thousand for this code.

    # Raise exception if temperature is zero
    # if notemperature == 0:
    #    raise SMRTError('Temperature used in calculation of ice permittivity is zero')

    # Issue warning if temperature is below 240K (real part is for T > 240K)
    # if np.any(temperature < 240):
    #    warnings.warn('Warning: temperature is below 240K. Ice permittivity is out of range of applicability')

    # Real part: from Mätzler and Wegmuller (1987)
    real_permittivity_ice = 3.1884 + 9.1e-4 * (temperature - 273.0)

    # Imaginary part: from Mätzler (2006)
    # NB frequencies in original equations are in GHz, here in Hz.
    freqGHz = frequency * 1e-9
    theta = (300.0 / temperature) - 1.0  # Floats needed for correct calculation in py2.7 but not needed in 3.x
    alpha = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)
    beta = (0.0207 / temperature) * (np.exp(335.0 / temperature) / (np.exp(335.0 / temperature) - 1.0)**2) + (
        1.16e-11 * (freqGHz)**2 + np.exp(-9.963 + 0.0372 * (temperature - 273.0)))
    # Salinity modifications from equations 5.36 and 5.37 in Mätzler (2006)
    salinity_effect = 1866.0 * np.exp(-0.317 * freqGHz) + (72.2 + 6.02 * freqGHz) * (273.16 - temperature)
    imag_permittivity_ice = alpha / freqGHz + beta * freqGHz + salinity / (0.013 * salinity_effect)

    return real_permittivity_ice + 1j * imag_permittivity_ice


@layer_properties("temperature")
def ice_permittivity_hufford91_maetzler87(frequency, temperature):
    """ Calculates the complex ice dielectric constant depending on the frequency and temperature.
    Real part of permittivity follows Mätzler and Wegmuller (1987): 
    Dielectric properties of freshwater ice at microwave frequencies, 10.1088/0022-3727/20/12/013.
    The imaginary part is based on Hufford 1991: 
    A model for the complex permittivity of ice at frequencies below 1 THz, 10.1007/BF01008898. 
    The Hufford model is derived for frequencies up to 1000 GHz and temperatures from -40°C to 0°C.
    This gives exact agreement with the MEMLS_ice model version used in Rückert et al., 2023.
    
    :param frequency: Frequency in Hz
    :param temperature: ice temperature in K
    :returns: complex permittivity of pure ice
    """

    # Raise exception if temperature is zero
    if np.any(temperature > FREEZING_POINT):
        raise SMRTError(f"The ice temperature must be lower or equal to {FREEZING_POINT}K")

    # Real part: from Mätzler and Wegmuller (1987)
    real_permittivity_ice = 3.1884 + 9.1e-4 * (temperature - 273.0)

    # frequencies in original equations are in GHz, here in Hz.
    freqGHz = frequency * 1e-9
    # Equations 4, 6,7,11 in Hufford 1991:
    theta = 300 / temperature - 1.0
    alpha = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)
    beta = ((0.502 - 0.131 * theta) / (1 + theta)) * 1e-4 + (0.542e-6 * ((1 + theta) /
         (theta + 0.0073))**2)
    imag_permittivity_ice = (alpha / freqGHz) + (beta * freqGHz)

    return real_permittivity_ice + 1j * imag_permittivity_ice
