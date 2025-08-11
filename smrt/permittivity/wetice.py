# coding: utf-8
"""
Provide equations to compute the effective permittivity of wet ice.
"""
import numpy as np
from ..core.layer import layer_properties
from .ice import ice_permittivity_maetzler06
from .water import water_permittivity_maetzler87
from .generic_mixing_formula import maxwell_garnett_for_spheres, polder_van_santen


@layer_properties("temperature", "liquid_water")
def wetice_permittivity_bohren83(frequency, temperature, liquid_water):
    """
    Calculate the dielectric constant of wet particules of ice using Maxwell Garnet equation using water as the background and 
    ice as the inclusions. As reported by Bohren and Huffman 1983 according to Ya Qi Jin (1993), eq 8-69, p282.

    Note:
        See also Chopra and Reddy (1986).

    Args:
        frequency: Frequency in Hz.
        temperature: Temperature in K.
        liquid_water: Fractional volume of water with respect to ice+water volume.

    Returns:
        Complex permittivity of pure ice.

    References:
         Bohren, C. F., Huffman, D. R. (1983). Absorption and scattering of light by small particles. New York, Wiley-Interscience, 
         1983, 541 p.

         Chopra, K.L., Reddy, G.B. Optically selective coatings. Pramana - J Phys 27, 193–217 (1986). https://doi.org/10.1007/BF02846338

         Jin, Y.Q., 1993. Electromagnetic scattering modelling for quantitative remote sensing. World Scientific.
    """

    epsice = ice_permittivity_maetzler06(frequency, temperature)

    if np.all(liquid_water <= 0.0):
        return epsice

    epswater = water_permittivity_maetzler87(frequency, temperature)

    return maxwell_garnett_for_spheres(1 - liquid_water, epswater, epsice)


@layer_properties("temperature", "liquid_water")
def symmetric_wetice_permittivity(frequency, temperature, liquid_water):
    """
    Calculate the dielectric constant of wet particules of ice using Polder van Santen Maxwell equation 
    assuming both ice and water are fully mixed. 
    
    Note:
        This applies to intermediate content of wetness. Typically liquid_water=0.5.

    Args:
        frequency: Frequency in Hz.
        temperature: Temperature in K.
        liquid_water: Fractional volume of water with respect to ice+water volume.

    Returns:
        Complex permittivity of pure ice.

    """

    epsice = ice_permittivity_maetzler06(frequency, temperature)

    if np.all(liquid_water <= 0.0):
        return epsice

    epswater = water_permittivity_maetzler87(frequency, temperature)

    return polder_van_santen(liquid_water, epsice, epswater)
