import warnings

import numpy as np

from smrt.core.globalconstants import FREEZING_POINT, GHz
from smrt.permittivity.ice import ice_permittivity_maetzler06
from ..core.layer import layer_properties


@layer_properties("temperature", "salinity")
def impure_ice_permittivity_maetzler06(frequency, temperature, salinity):
    """ Computes permittivity of impure ice from Maetzler 2006 - Thermal Microwave Radiation: Applications for Remote Sensing

        Model developed for salinity around 0.013 PSU, so it is not recommended for sea ice
        :param temperature: ice temperature in K
        :param salinity: salinity of ice in kg/kg (see PSU constant in smrt module)

    **Usage example**::

        from smrt.permittivity.saline_ice import impure_ice_permittivity_maetzler06
        eps_ice = impure_ice_permittivity_maetzler06(frequency=18e9, temperature=270, salinity=0.013)

    """

    # Issue warning if salinity > 0.013 PSU
    if salinity > 0.013e-3:
        warnings.warn(
            "This permittivity model was developed for saline impurities of around 0.013 10^-3 kg/kg (or 0.013 PSU)")

    # Modify imaginary component calculated for pure ice
    pure_ice_permittivity = ice_permittivity_maetzler06(frequency, temperature)

    # Equation 5.37 from Maetzler 2006: Thermal Microwave Radiation: Applications for Remote Sensing
    freqGHz = frequency / GHz
    g0 = 1866 * np.exp(-0.317 * freqGHz)
    g1 = 72.2 + 6.02 * freqGHz

    # Equation 5.36
    delta_Eimag = 1. / (g0 + g1 * (FREEZING_POINT - temperature))

    # Equation 5.38
    S0 = 0.013  # CURRENT UNITS ARE PSU
    # print (pure_ice_permittivity, delta_Eimag * salinity * 1e3 / S0)
    # print (pure_ice_permittivity + 1j * delta_Eimag * salinity * 1e3 / S0)
    return pure_ice_permittivity + 1j * delta_Eimag * salinity * 1e3 / S0
