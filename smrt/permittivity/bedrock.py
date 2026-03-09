"""This module implements various bedrock dielectric constant models.
You can use the different permittivty in the permittivity_model argument of the `make_bedrock function
by calling them without the 'bedrock_permittivity_' prefix.

References:
- Hartlieb, P., Gloaguen, R., & Zimmermann, R. (2016). Dielectric properties of common rocks at microwave frequencies
   and their relevance for ground-penetrating radar. Engineering Geology, 204, 26-38, doi:10.1016/j.mineng.2015.11.008
 - Tulaczyk, S., & Foley, N. M. (2020). A compilation of geophysical properties of subglacial materials to inform
   interpretation of geophysical data from ice-covered regions. The Cryosphere, 14(11), 4495-4513,
   doi:10.5194/tc-14-4495-2020
 - Christianson, K., Peters, L. E., Tulaczyk, S., Mikesell, T. D., & Holschuh, N. (2016). Geophysical imaging of
   englacial and subglacial englacial and subglacial features of the western Greenland Ice Sheet. Journal of Geophysical
   Research: Earth Surface, 121(3), 492-514, doi:10.1002/2015JF003806

Example::

    from smrt.inputs.make_bedrock import make_bedrock
    bedrock = make_bedrock("flat", "granite_hartlieb16", temperature=270)

"""
import numpy as np
from smrt.core.globalconstants import PERMITTIVITY_OF_FREE_SPACE
from smrt.core.layer import layer_properties

@layer_properties()
def bedrock_permittivity_granite_hartlieb16(frequency, temperature):
    """
    Hartlieb et al. (2016) 10.1016/j.mineng.2015.11.008
    Frequency = 2450 MHz, temperature = 20 degC
    """  
    return 5.45 + 0.038j

@layer_properties()
def bedrock_permittivity_basalt_hartlieb16(frequency, temperature):
    """
    Hartlieb et al. (2016) 10.1016/j.mineng.2015.11.008
    Frequency = 2450 MHz, temperature = 20 degC
    """  
    return 7.67 + 0.270j

@layer_properties()
def bedrock_permittivity_sandstone_hartlieb16(frequency, temperature):
    """
    Hartlieb et al. (2016) 10.1016/j.mineng.2015.11.008
    Frequency = 2450 MHz, temperature = 20 degC
    """  
    return 7.67 + 0.081j

@layer_properties()
def bedrock_permittivity_frozen_bedrock_tulaczyk20(frequency, temperature):
    """
    Tulaczyk & Foley (2020) 10.5194/tc-14-4495-2020
    Frequency = 5e6HZ, temperature = close to 0 degC
    """  
    angular_frequency = 2 * np.pi * frequency
    return 2.7 + 1j * (0.0002 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_saturated_bedrock_tulaczyk20(frequency, temperature):
    """
    Tulaczyk & Foley (2020) 10.5194/tc-14-4495-2020
    Frequency = Midpoints used, conductivity measurment frequency 0.9 to 25 kHZ,
    temperature = close to 0 degC
    """  
    angular_frequency = 2 * np.pi * frequency
    return 9.5 + 1j * (0.0055 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_sandy_till_tulaczyk20(frequency, temperature):
    """
    Tulaczyk & Foley (2020) 10.5194/tc-14-4495-2020
    Frequency = Midpoint & Upper bound, unclear frequency in paper,
    temperature = close to 0 degC
    """  
    angular_frequency = 2 * np.pi * frequency
    return 13.0 + 1j * (0.02 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_fairbanks_silt_tulaczyk20(frequency, temperature):
    """
    Tulaczyk & Foley (2020) 10.5194/tc-14-4495-2020
    Frequency = 100MHz,
    temperature = close to 0 degC
    """  
    angular_frequency = 2 * np.pi * frequency
    return 24.0 + 1j * (0.043 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))


@layer_properties()
def bedrock_permittivity_clay_bearing_till_tulaczyk20(frequency, temperature):
    """
    Tulaczyk & Foley (2020) 10.5194/tc-14-4495-2020
    Frequency = Midpoints used, unclear frequency in paper,
    temperature = close to 0 degC
    """  
    angular_frequency = 2 * np.pi * frequency
    return 13.0 + 1j * (0.0575 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_clay_tulaczyk20(frequency, temperature):
    """
    Tulaczyk & Foley (2020) 10.5194/tc-14-4495-2020
    Frequency =  100MHz,
    temperature = close to 0 degC
    """  
    angular_frequency = 2 * np.pi * frequency
    return 31.0 + 1j * (0.24 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_marine_clay_tulaczyk20(frequency, temperature):
    """
    Tulaczyk & Foley (2020) 10.5194/tc-14-4495-2020
    Frequency =  Midpoint used, unclear frequency in paper,
    temperature = close to 0 degC
    """  
    angular_frequency = 2 * np.pi * frequency
    return 31.0 + 1j * (0.55 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_debris_laden_ice_christianson16(frequency, temperature):
    """
    Christianson et al. (2016) 10.1002/2015JF003806
    Frequency =  5MHz,
    temperature = unclear in paper, depend of medium
    """  
    angular_frequency = 2 * np.pi * frequency
    return 3.1 + 1j * (8.0e-5 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_sand_christianson16(frequency, temperature):
    """
    Christianson et al. (2016) 10.1002/2015JF003806
    Frequency =  5MHz,
    temperature = unclear in paper, depend of medium
    """  
    angular_frequency = 2 * np.pi * frequency
    return 2.6 + 1j * (1.3e-4 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_groundwater_till_christianson16(frequency, temperature):
    """
    Christianson et al. (2016) 10.1002/2015JF003806
    Frequency =  5MHz,
    temperature = unclear in paper, depend of medium
    """  
    angular_frequency = 2 * np.pi * frequency
    return 36.0 + 1j * (0.037 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_freshwater_till_christianson16(frequency, temperature):
    """
    Christianson et al. (2016) 10.1002/2015JF003806
    Frequency =  5MHz,
    temperature = unclear in paper, depend of medium
    """  
    angular_frequency = 2 * np.pi * frequency
    return 13.0 + 1j * (2.5e-4 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_frozen_till_christianson16(frequency, temperature):
    """
    Christianson et al. (2016) 10.1002/2015JF003806
    Frequency =  5MHz,
    temperature = unclear in paper, depend of medium
    """      
    angular_frequency = 2 * np.pi * frequency
    return 2.9 + 1j * (3.4e-4 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_frozen_bedrock_christianson16(frequency, temperature):
    """
    Christianson et al. (2016) 10.1002/2015JF003806
    Frequency =  5MHz,
    temperature = unclear in paper, depend of medium
    """      
    angular_frequency = 2 * np.pi * frequency
    return 2.7 + 1j * (2.0e-4 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))

@layer_properties()
def bedrock_permittivity_unfrozen_bedrock_christianson16(frequency, temperature):
    """
    Christianson et al. (2016) 10.1002/2015JF003806
    Frequency =  5MHz,
    temperature = unclear in paper, depend of medium
    """      
    angular_frequency = 2 * np.pi * frequency
    return 12.0 + 1j * (0.0048 / (angular_frequency * PERMITTIVITY_OF_FREE_SPACE))
