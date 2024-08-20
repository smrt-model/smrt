"""These permittivity formulations are for use with saline ice, and possibly in some case with saline snow. See also saline_snow.py in the latter case.
"""


import warnings

import numpy as np

from smrt.core.globalconstants import FREEZING_POINT, GHz
from smrt.permittivity.ice import ice_permittivity_maetzler06
from smrt.permittivity.saline_water import brine_permittivity_stogryn85
from smrt.permittivity.generic_mixing_formula import polder_van_santen
from smrt.core.layer import layer_properties


@layer_properties("temperature", "salinity")
def impure_ice_permittivity_maetzler06(frequency, temperature, salinity):
    """ Computes permittivity of impure ice from Maetzler 2006 - Thermal Microwave Radiation: Applications for Remote Sensing.
     Model developed for salinity around 0.013 PSU. The extrapolation is based on linear assumption to salinity, so it is not recommended for much higher salinity.

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


@layer_properties("temperature", "brine_volume_fraction",
                  optional=("brine_inclusion_shape", "brine_mixing_ratio", "ice_permittivity_model", "brine_permittivity_model"))
def saline_ice_permittivity_pvs_mixing(frequency, temperature, brine_volume_fraction, brine_inclusion_shape='spheres',
                                       brine_mixing_ratio=1, ice_permittivity_model=None, brine_permittivity_model=None):
    """Computes effective permittivity of saline ice using the Polder Van Santen mixing formulaes.

        :param frequency: frequency in Hz
        :param temperature: ice temperature in K
        :param brine_volume_fraction: brine / liquid water fraction in sea ice
        :param brine_inclusion_shape: Assumption for shape(s) of brine inclusions. Can be a string for single shape,
            or a list/tuple/dict of strings for mixture of shapes. So far, we have the following shapes: "spheres" and
            "random_needles" (i.e. randomly-oriented elongated ellipsoidal inclusions).
            If the argument is a dict, the keys are the shapes and the values are the mixing ratio.
            If it is a list, the mixing_ratio argument is required.
        :param brine_mixing_ratio: The mixing ratio of the shapes. This is only relevant when inclusion_shape is a list/tuple.
            Mixing ratio must be a sequence with length len(inclusion_shape)-1.
            The mixing ratio of the last shapes is deduced as the sum of the ratios must equal to 1.
        :param ice_permittivity_model: pure ice permittivity formulation (default is ice_permittivity_matzler87)
        :param brine_permittivity_model: brine permittivity formulation (default is brine_permittivity_stogryn85)
    """

    if ice_permittivity_model is None:
        ice_permittivity_model = ice_permittivity_maetzler06  # default ice permittivity model

    if brine_permittivity_model is None:
        brine_permittivity_model = brine_permittivity_stogryn85  # default brine permittivity model

    pure_ice_permittivity = ice_permittivity_model(frequency, temperature)
    brine_permittivity = brine_permittivity_model(frequency, temperature)

    return polder_van_santen(brine_volume_fraction,
                             e0=pure_ice_permittivity,
                             eps=brine_permittivity,
                             inclusion_shape=brine_inclusion_shape, mixing_ratio=brine_mixing_ratio)
