# coding: utf-8

"""This module provides a function to build bedrock model using rock permittivity published in the literature:
Hartlieb et al. (2016), Tulaczyk and Foley (2020), and Christianson et al. (2016). The list of available bedrock
permittivity models can be obtained using the function `list_bedrock_permittivity_models
()`.

Example::

    from smrt import make_bedrock, list_bedock_permittivity_models

    print("Available bedrock permittivity models:", list_bedrock_permittivity_models())

    bedrock = make_bedrock("rough_choudhury79", "granite_hartlieb16", temperature=273.15, roughness_rms=0.01, corr_length=0.1)

References:
 - Hartlieb, P., Gloaguen, R., & Zimmermann, R. (2016). Dielectric properties of common rocks at microwave frequencies
   and their relevance for ground-penetrating radar. Engineering Geology, 204, 26-38, doi:10.1016/j.mineng.2015.11.008
 - Tulaczyk, S., & Foley, N. M. (2020). A compilation of geophysical properties of subglacial materials to inform
   interpretation of geophysical data from ice-covered regions. The Cryosphere, 14(11), 4495-4513,
   doi:10.5194/tc-14-4495-2020
 - Christianson, K., Peters, L. E., Tulaczyk, S., Mikesell, T. D., & Holschuh, N. (2016). Geophysical imaging of
   englacial and subglacial englacial and subglacial features of the western Greenland Ice Sheet. Journal of Geophysical
   Research: Earth Surface, 121(3), 492-514, doi:10.1002/2015JF003806
"""

from numbers import Number
from typing import Callable, Dict, Tuple, Union

import numpy as np

from smrt.core.error import SMRTError
from smrt.core.globalconstants import PERMITTIVITY_OF_FREE_SPACE
from smrt.core.interface import SubstrateBase, get_substrate_model

# --- DATA TABLES ---

_BEDROCK_COMPLEX_PERMITTIVITY_DATA_TABLE: Dict[str, Tuple[complex, float]] = {
    # This tabe handle two formating:
    # Pure complex permivitivty format (hartlieb) or tuple of real permivity and conductivity (tulaczyk)
    # If conductivity is non-zero, the imaginary part of complex_permittivity must be zero !
    # Hartlieb et al. (2016) 10.1016/j.mineng.2015.11.008
    # Data: {rock_type: (real_permittivity + imaginary_permittivity*j, no conductivity)}
    # Frequency = 2450 MHz, temperature = 20 degc
    "granite_hartlieb16": (5.45 + 0.038j, 0),  # data give complex permitivity for 2450MHZ, no conductivity needed
    "basalt_hartlieb16": (7.67 + 0.270j, 0),  # data give complex permitivity for 2450MHZ, no conductivity needed
    "sandstone_hartlieb16": (7.67 + 0.081j, 0),  # data give complex permitivity for 2450MHZ, no conductivity needed
    # Tulaczyk and Foley (2020) 10.5194/tc-14-4495-2020
    # Data from Table 1: {material_type: (real permittivity, conductivity)}
    # Frequency: various (tens to hundreads of MHZ), see legend of Table 1, temperature : close to 0°c whenever possible
    #'glacier_ice_tulaczyk20': (3.2+0j, 0.00007),      # conductivity measurment frequency 5e6HZ
    "frozen_bedrock_tulaczyk20": (2.7 + 0j, 0.0002),  # conductivity measurment frequency 5e6HZ
    #'marine_ice': (3.4+0j, 0.0003),                   # conductivity measurment frequency 150e6HZ
    "saturated_bedrock_tulaczyk20": (
        9.5 + 0j,
        0.0055,
    ),  # Midpoints used, conductivity measurment frequency 0.9 to 25 kHZ
    #'saline_basal_ice_tulaczyk20': (3.4+0j, 0.02),    # Upper bound conductivity measurment frequency 150e6 HZ
    "sandy_till_tulaczyk20": (13.0 + 0j, 0.02),  # Midpoint & Upper bound, unclear frequency in paper
    #'subglacial_water_tulaczyk20': (88.0+0j, 0.04 ),  # no freqeuncy given in paper
    "fairbanks_silt_tulaczyk20": (24.0 + 0j, 0.043),  # conductivity measurment frequency 100MHZ
    "clay-bearing_till_tulaczyk20": (13.0 + 0j, 0.0575),  # Midpoints used, unclear frequency in paper
    "clay_tulaczyk20": (31.0 + 0j, 0.24),  # conductivity measurment frequency 100MHZ
    "marine_clay_tulaczyk20": (31.0 + 0j, 0.55),  # Midpoint used, unclear frequency in paper
    #'seawater_tulaczyk20': (79.0+0j, 2.90 ),          # unclear frequency in paper
    #'brine_tulaczyk20': (62.0+0j, 4.89 ),             # unclear frequency in paper
    # Christianson et al. (2016) 10.1002/2015JF003806
    # Data: {material_type: (real permittivity, conductivity)}
    # Frequency = 5MHZ, temperature ? (depend of medium)
    #'glacier_ice_christianson16': (3.2+0j, 7.0e-5),    # data give real permittivity and conductivity for 5MHZ
    #'marine_ice_christianson16': (3.4+0j, 5.7e-4),
    #'groundwater_ice_christianson16': (3.2+0j, 6.6e-4),
    "debris-laden_ice_christianson16": (3.1 + 0j, 8.0e-5),  # data give real permittivity and conductivity for 5MHZ
    #'seawater_christianson16': (77.0+0j, 2.9),
    #'freshwater_christianson16': (80.0+0j, 1.0e-4),
    #'groundwater_christianson16': (80.0+0j, 0.37),
    "sand_christianson16": (2.6 + 0j, 1.3e-4),
    "groundwater_till_christianson16": (36.0 + 0j, 0.037),  # data give real permittivity and conductivity for 5MHZ
    "freshwater_till_christianson16": (13.0 + 0j, 2.5e-4),  # Midpoints used
    "frozen_till_christianson16": (2.9 + 0j, 3.4e-4),
    "frozen_bedrock_christianson16": (2.7 + 0j, 2.0e-4),
    "unfrozen_bedrock_christianson16": (12.0 + 0j, 0.0048),  # data give real permittivity and conductivity for 5MHZ
}


# --- PUBLIC FUNCTIONS (make_bedrock, ...) ---


def list_bedock_permittivity_models():
    """
    List the available bedrock permittivity models defined in the internal database.
    """

    return list(_BEDROCK_COMPLEX_PERMITTIVITY_DATA_TABLE.keys())


def make_bedrock(
    substrate_model: Union[str, type[SubstrateBase]],
    bedrock_permittivity_model: Union[
        str, complex, Number, Callable
    ],  # TODO: improve typing to accept ndarray generitically
    temperature: float,
    **kwargs,
) -> SubstrateBase:
    """
    Construct a bedrock substrate model based on a given substrate model,
    bedrock permittivity model and temperature.

    :param substrate_model: The electromagnetic surface model to apply to the
        bedrock. This can be a string (e.g., "flat", "rough_choudhury79")
        referring to a predefined model, or a direct `SubstrateBase` class.
    :param bedrock_permittivity_model: Defines the dielectric properties of the
        bedrock. This can be specified in different ways:
        -   **A string name**: Refers to a predefined bedrock type from a
            geophysical database (e.g., "granite_hartlieb16",
            "frozen_bedrock_tulaczyk20"). These models include complex
            permittivity and/or conductivity values derived from scientific
            literature, suitable for various rock types and conditions.
            You can list available models using `list_bedrock_permittivity_models()`.
        -   **A constant complex number**: For a fixed, frequency- and
            temperature-independent permittivity (e.g., `3.15 + 0.001j`).
        -   **A callable function**: A function with the signature
            `(frequency_Hz: float, temperature: float) -> complex`.
    :param temperature: The physical temperature of the bedrock in Kelvin (K).
        This parameter is mandatory.
    :param kwargs: Additional keyword arguments.
    :raises SMRTError: If `temperature` is not provided, if
        `bedrock_permittivity_model` is an unrecognized string (not found in
        the predefined list), or if other required parameters for the chosen
        substrate model are missing or invalid.
    :return: An initialized instance of a `SubstrateBase` model, representing
        the bedrock with its specified electromagnetic and physical properties.
        This object can then be used in radiative transfer simulations.
    """
    # 1. Select the bedrock permittivity model.

    if isinstance(bedrock_permittivity_model, str):
        try:
            permittivity_model = _bedrock_permittivity_model_from_table(bedrock_permittivity_model)
        except KeyError:
            raise SMRTError(
                f"The bedrock permittivity model name '{bedrock_permittivity_model}' is not recognized. "
                f"Choose from: {', '.join(list_bedock_permittivity_models())}, a complex number, or a callable function."
            )

    # 2. Handle constant number or callable permittivity model
    else:
        permittivity_model = bedrock_permittivity_model

    # 3. Process the substrate_model argument (assumes get_substrate_model returns a class/type)
    if isinstance(substrate_model, str):
        SubstrateClass = get_substrate_model(substrate_model)
    else:
        SubstrateClass = substrate_model

    # create the instance
    return SubstrateClass(permittivity_model=permittivity_model, temperature=temperature, **kwargs)


# --- PRIVATE FUNCTIONS ---


def _bedrock_permittivity_model_from_table(bedrock_permittivity_model: str) -> Callable:
    permittivity_value, conductivity = _BEDROCK_COMPLEX_PERMITTIVITY_DATA_TABLE[bedrock_permittivity_model]

    def permittivity_model(frequency, permittivity_value=permittivity_value, conductivity=conductivity, **kwargs):
        angular_frequency = 2 * np.pi * frequency
        return permittivity_value + 1j * (conductivity / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

    return permittivity_model
