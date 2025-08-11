# coding: utf-8
"""
Provide equations to compute the effective permittivity of wet ice.

The wetsnow module is to be removed in a future version. It has been renamed wetice because the naming was ambiguous.
"""
import numpy as np
from ..core.layer import layer_properties
from .ice import ice_permittivity_maetzler06
from .water import water_permittivity

import warnings


warnings.warn("The wetsnow module is to be removed in a future version."
              " It has been renamed wetice because the naming was ambiguous.", DeprecationWarning)


@layer_properties("temperature", "liquid_water")
def wetsnow_permittivity(frequency, temperature, liquid_water):
    """
    Calculate the dielectric constant of wet particles of ice using Bohren and Huffman (1983) according to Ya Qi Jin (1996), 
    eq 8-69, p282.
    
    Note:
        See also Chopra and Reddy (1986).

    Args:
        frequency: frequency in Hz
        temperature: temperature in K
        liquid_water: fractional volume of water with respect to ice+water volume

    Returns:
        complex permittivity of pure ice

    References:
         Bohren, C. F., Huffman, D. R. (1983). Absorption and scattering of light by small particles. New York, Wiley-Interscience, 
         1983, 541 p.

        Chopra, K.L., Reddy, G.B. Optically selective coatings. Pramana - J Phys 27, 193–217 (1986). https://doi.org/10.1007/BF02846338

         Jin, Y.Q., 1993. Electromagnetic scattering modelling for quantitative remote sensing. World Scientific.
    """
    # from http://books.google.com/books?id=Y_-113zIvgkC&pg=PA142&lpg=PA142&dq=effective+dielectric+constant+sphere+coated+small&source=bl&ots=ZVfwvkA0K1&sig=P7fHb0Jff8C-7-GrlEnWRZkkxY8&hl=en&ei=RHfDTrmjJYXj8AO3v7ScCw&sa=X&oi=book_result&ct=result&resnum=3&ved=0CDYQ6AEwAg#v=onepage&q=effective%20dielectric%20constant%20sphere%20coated%20small&f=false
    # see also: K L CHOPRA and G B REDDY, Praman.a- Optically selective coatings, J. Phys., Vol. 27, Nos 1 & 2, July & August 1986, pp. 193-217.

    warnings.warn("The wetsnow_permittivity function is to be removed in a future version."
                  " It is replaced by wetice_permittivity in the module wetice", DeprecationWarning)

    epsice = ice_permittivity_maetzler06(frequency, temperature)

    if np.all(liquid_water <= 0.0):
        return epsice

    epswater = water_permittivity(frequency, temperature)

    S = 1 - liquid_water

    Cplus = epsice + 2 * epswater
    Cminus = (epsice - epswater) * S

    Ewi = (Cplus + 2 * Cminus) / (Cplus - Cminus) * epswater

    return Ewi
