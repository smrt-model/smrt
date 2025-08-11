# coding: utf-8
"""
Provide equations to compute the effective permittivity of water.
"""
import numpy as np

from ..core.layer import layer_properties
from ..core.globalconstants import FREEZING_POINT, GHz
from ..core.error import SMRTError


@layer_properties("temperature")
def water_permittivity_maetzler87(frequency, temperature):
    """
    Calculate the complex water dielectric constant depending on the frequency and temperature based on Mätzler & Wegmuller (1987).

    Args:
        frequency: Frequency in Hz.
        temperature: Temperature in K.

    Raises:
        Exception: If liquid water > 0 or salinity > 0 (model unsuitable).

    Returns:
        complex: Complex permittivity of pure ice.
    
    References:
        Matzler, C., Wegmuller, U., (1987) Dielectric properties of freshwater ice at microwave frequencies. J. Phys. D: 
        Appl. Phys, 20, 1623–1630, https://doi.org/10.1088/0022-3727/20/12/013.
    """
    if temperature < FREEZING_POINT:
        raise SMRTError(f"The water temperature must be higher or equal to {FREEZING_POINT}K")

    freqGHz = frequency / 1e9

    theta = 1 - 300.0 / temperature

    e0 = 77.66 - 103.3 * theta
    e1 = 0.0671 * e0

    f1 = 20.2 + 146.4 * theta + 316 * theta**2
    e2 = 3.52 + 7.52 * theta
    #  % version of Liebe MPM 1993 uses: e2=3.52
    f2 = 39.8 * f1

    Ew = e2 + (e1 - e2) / complex(1, -freqGHz / f2) + (e0 - e1) / complex(1, -freqGHz / f1)

    return Ew


# for backward compatibility
water_permittivity = water_permittivity_maetzler87


@layer_properties("temperature")
def water_permittivity_tiuri80(frequency, temperature):
    """
    Calculate the complex water dielectric constant reported by Tiuri and Schultz (1980).

    Args:
        frequency: Frequency in Hz.
        temperature: Temperature in K.

    Raises:
        SMRTError: If the water temperature is below the freezing point.

    Returns:
        complex: Complex permittivity of water.

    References:
        Tiuri, M. and Schultz, H., Theoretical and experimental studies of microwave radiation from a natural snow field. In Rango, A. , ed.
        Microwave remote sensing of snowpack properties. Proceedings of a workshop ... Fort Collins, Colorado, May 20-22, 1980. 
        Washington, DC, National Aeronautics and Space Center, 225-234. (Conference Publication 2153.)
        https://ntrs.nasa.gov/api/citations/19810010984/downloads/19810010984.pdf
    """
    freqGHz = frequency / GHz

    tempC = temperature - FREEZING_POINT

    if np.any(tempC < 0):
        raise SMRTError(f"The water temperature must be higher or equal to {FREEZING_POINT}K")

    e2 = 4.903e-2

    e1 = 87.74 - 0.4008 * tempC + 9.398e-4 * tempC**2 + 1.410e-6 * tempC**3

    # version of Liebe 1991 because Tiuri 1980 does not give the relaxation frequency
    theta = 1 - 300.0 / temperature
    f1 = 20.2 + 146.4 * theta + 316 * theta**2


    Ew = e2 + (e1 - e2) / complex(1, -freqGHz /f1)

    return Ew


