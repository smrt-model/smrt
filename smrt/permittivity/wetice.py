# coding: utf-8

import numpy as np
from ..core.layer import layer_properties
from .ice import ice_permittivity_maetzler06
from .water import water_permittivity_maetzler87
from .generic_mixing_formula import maxwell_garnett_for_spheres, polder_van_santen


@layer_properties("temperature", "liquid_water")
def wetice_permittivity_bohren83(frequency, temperature, liquid_water):
    """ calculate the dielectric constant of wet particules of ice using Maxwell Garnet equation using water as the background and 
    ice as the inclusions. As reported by Bohren and Huffman 1983 according to Ya Qi Jin, eq 8-69, 1996 p282

    see also: K L CHOPRA and G B REDDY, Praman.a- Optically selective coatings, J. Phys., Vol. 27, Nos 1 & 2, July & August 1986, pp. 193-217.

    :param frequency: frequency in Hz
    :param temperature: temperature in K
    :param liquid_water (fractional volume of water with respect to ice+water volume).
    :returns: Complex permittivity of pure ice

    """

    epsice = ice_permittivity_maetzler06(frequency, temperature)

    if np.all(liquid_water <= 0.0):
        return epsice

    epswater = water_permittivity_maetzler87(frequency, temperature)

    return maxwell_garnett_for_spheres(1 - liquid_water, epswater, epsice)


@layer_properties("temperature", "liquid_water")
def symmetric_wetice_permittivity(frequency, temperature, liquid_water):
    """ calculate the dielectric constant of wet particules of ice using Polder van Santen Maxwell equation 
    assuming both ice and water are fully mixed. This applies to intermediate content of wetness. Typically liquid_water=0.5.

    :param frequency: frequency in Hz
    :param temperature: temperature in K
    :param liquid_water (fractional volume of water with respect to ice+water volume).
    :returns: Complex permittivity of pure ice

    """

    epsice = ice_permittivity_maetzler06(frequency, temperature)

    if np.all(liquid_water <= 0.0):
        return epsice

    epswater = water_permittivity_maetzler87(frequency, temperature)

    return polder_van_santen(liquid_water, epsice, epswater)
