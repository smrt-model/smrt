# coding: utf-8

from __future__ import print_function

# Stdlib import
import math

# other import
# import numpy as np

# local import
# from ..core.error import SMRTError
from ..core.globalconstants import FREEZING_POINT
from ..core.layer import required_layer_properties

#
# for developers: see note in __init__.py
#


@required_layer_properties("temperature")
def ice_permittivity_matzler87(frequency, temperature):
    """ Calculates the complex ice dielectric constant depending on the frequency and temperature

    Based on Mätzler, C., & Wegmuller, U. (1987). Dielectric properties of freshwater 
    ice at microwave frequencies. *Journal of Physics D: Applied Physics*, 20(12), 1623-1630.
    This is the default model used in smrt.core.layer.make_snow_layer().

    :param frequency: frequency in Hz
    :param temperature: temperature in K
    :returns: Complex permittivity of pure ice

    **Usage example**::

        from smrt.permittivity.ice import ice_permittivity_matzler87
        eps_ice = ice_permittivity_matzler87(frequency=18e9, temperature=270, liquid_water=0, salinity=0)

    .. note::

        Ice permittivity is automatically calculated in smrt.core.layer.make_snow_layer() and
        is not set by the electromagnetic model module. An alternative
        to ice_permittivity_matzler87 may be specified as an argument to the make_snow_layer 
        function. The usage example is provided for external reference or testing purposes.

"""

    freqGHz = frequency / 1e9

    Ereal = 3.1884 + 9.1e-4 * (temperature - FREEZING_POINT)

    theta = 300.0 / temperature - 1.0
    alpha = (0.00504 + 0.0062 * theta) * math.exp(-22.1 * theta)

    B1 = 0.0207
    B2 = 1.16e-11
    b = 335.
    deltabeta = math.exp(- 9.963 + 0.0372 * (temperature - FREEZING_POINT))
    betam = (B1 / temperature) * (math.exp(b / temperature) / ((math.exp(b / temperature) - 1)**2)) + B2 * freqGHz**2
    beta = betam + deltabeta

    Eimag = alpha / freqGHz + beta * freqGHz

    return Ereal + 1j * Eimag


@required_layer_properties("temperature")
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
    real_permittivity_ice = 3.1884 + 9.1e-4 * (temperature - 273.0)

    # Imaginary part: from Mätzler (2006)
    # NB frequencies in original equations are in GHz, here in Hz.
    freqGHz = frequency * 1e-9
    theta = (300.0 / temperature) - 1.0  # Floats needed for correct calculation in py2.7 but not needed in 3.x
    alpha = (0.00504 + 0.0062 * theta) * math.exp(-22.1 * theta)
    beta = (0.0207 / temperature) * (math.exp(335.0 / temperature) / (math.exp(335.0 / temperature) - 1.0)**2) + (
        1.16e-11 * (freqGHz)**2 + math.exp(-10.02 + 0.0364 * (temperature - 273.0)))
    imag_permittivity_ice = alpha / freqGHz + beta * freqGHz

    return real_permittivity_ice + 1j * imag_permittivity_ice


@required_layer_properties("temperature")
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
    alpha = (0.00504 + 0.0062 * theta) * math.exp(-22.1 * theta)
    beta = (0.0207 / temperature) * (math.exp(335.0 / temperature) / (math.exp(335.0 / temperature) - 1.0)**2) + (
        1.16e-11 * (freqGHz)**2 + math.exp(-9.963 + 0.0372 * (temperature - 273.16)))
    imag_permittivity_ice = alpha / freqGHz + beta * freqGHz

    return real_permittivity_ice + 1j * imag_permittivity_ice


@required_layer_properties("temperature", "salinity")
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
    alpha = (0.00504 + 0.0062 * theta) * math.exp(-22.1 * theta)
    beta = (0.0207 / temperature) * (math.exp(335.0 / temperature) / (math.exp(335.0 / temperature) - 1.0)**2) + (
        1.16e-11 * (freqGHz)**2 + math.exp(-9.963 + 0.0372 * (temperature - 273.0)))
    # Salinity modifications from equations 5.36 and 5.37 in Mätzler (2006)
    salinity_effect = 1866.0 * math.exp(-0.317 * freqGHz) + (72.2 + 6.02 * freqGHz) * (273.16 - temperature)
    imag_permittivity_ice = alpha / freqGHz + beta * freqGHz + salinity / (0.013 * salinity_effect)

    return real_permittivity_ice + 1j * imag_permittivity_ice
