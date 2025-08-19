"""
Provide formaluations for the depolarization factors used with the Polden van Santen or Maxwell Garnett mixing formulations.
"""
from typing import Optional
import numpy as np


def depolarization_factors_spheroids(length_ratio: Optional[float]=None, **kwargs):
    """
    Calculate depolarization factors for use in effective permittivity models. These
    are a measure of the anisotropy of the snow. Default is spherical isotropy.

    Args:
        length_ratio: [Optional] ratio of microstructure length measurement in x/y direction to z-direction [unitless].

    Returns:
        [x, y, z] depolarization factor array.

    **Usage**::
    
        from smrt.permittivity.generic_mixing_formula import depolarization_factors
        #default depol of 1
        depol_xyz = depolarization_factors()
        #provide a length ratio
        depol_xyz = depolarization_factors(length_ratio=1.2)

    References:
        Mätzler, C. (1996). Microwave permittivity of dry snow. Geoscience and Remote Sensing, EEE Transactions on 
        Geoscience and Remote Sensing,, 34(2), 573–581. https://doi.org/10.1109/36.485133

        Löwe, H., Riche, F., and Schneebeli, M.: A general treatment of snow microstructure exemplified by an improved
        relation for thermal conductivity, The Cryosphere, 7, 1473–1480, https://doi.org/10.5194/tc-7-1473-2013, 2013.
    """

    # If a length ratio is not specified, assumes spherical isotropy
    if length_ratio is None:
        length_ratio = 1.0

    # Calculation of anisotropy factor
    if length_ratio == 1:
        anisotropy_q = 1.0 / 3.0
    elif length_ratio > 1:
        # Upper Equation 4 from Löwe et al. TC (2013)
        chi_b = np.sqrt(1.0 - 1.0 / (length_ratio**2.0))
        ln_term = np.log((1.0 + chi_b) / (1.0 - chi_b))
        anisotropy_q = 0.5 * (1.0 + (1.0 / (length_ratio**2.0 - 1.0)) * (1.0 - (1.0 / (2.0 * chi_b)) * ln_term))
    else:
        # Lower Equation 4 from Löwe et al. TC (2013)
        # Lower equation 7 in Matzler TGRS 1996
        chi_a = np.sqrt(1.0 / length_ratio**2.0 - 1.0)
        anisotropy_q = 0.5 * (1.0 + (1.0 / (length_ratio**2.0 - 1.0)) * (1.0 - (1.0 / chi_a) * np.arctan(chi_a)))

    return np.array([anisotropy_q, anisotropy_q, (1.0 - 2.0 * anisotropy_q)])


def depolarization_factors_matzler96(frac_volume: float, **kwargs):
    """
    Calculate depolarization factors with Mätzler (1996) for the Polden van Santen (Sihvola, 1999) permittivity model.

    Args:
        frac_volume: fractional volume of ice.

    Returns:
        [x, y, z] depolarization factor array.

    References:
        Mätzler, C. (1996). Microwave permittivity of dry snow. Geoscience and Remote Sensing, EEE Transactions on 
        Geoscience and Remote Sensing,, 34(2), 573–581. https://doi.org/10.1109/36.485133

        Sihvola, A.: Electromagnetic Mixing Formulas and Applications, 1999, INSTITUTION OF ENGINEERING & T
        http://www.ebook.de/de/product/21470462/a_sihvola_electromagnetic_mixing_formulas_and_applications.html
    """

    anisotropy_q = np.where(
        frac_volume < 0.33,
        01.0 + 0.5 * frac_volume,
        np.where(frac_volume < 0.71, 0.18 + 3.24 * (frac_volume - 0.49) ** 2, 1 / 3),
    )  # Eq 9
    return np.array([anisotropy_q, anisotropy_q, (1.0 - 2.0 * anisotropy_q)])


def depolarization_factors_oblate_matzler98(frac_volume: float, **kwargs):
    """
    Calculate depolarization factors with Mätzler (1998) for the Polden van Santen (Sihvola, 1999) permittivity model.

    Args:
        frac_volume: fractional volume of ice.
    
    Returns:
        [x, y, z] depolarization factor array.
        
    References:
        Mätzler, C. (1998). Improved Born approximation for scattering of radiation in a granular 
        medium. J. Appl. Phys., 83(11), 6111–6117. https://doi.org/10.1063/1.367496
    
        Sihvola, A.: Electromagnetic Mixing Formulas and Applications, 1999, INSTITUTION OF ENGINEERING & T
        http://www.ebook.de/de/product/21470462/a_sihvola_electromagnetic_mixing_formulas_and_applications.html
    """

    anisotropy_q = np.where(
        frac_volume < 0.33, 01.0 + 0.5 * frac_volume, np.where(frac_volume < 0.6, 0.476 - 0.64 * frac_volume, 0.092)
    )  # Eq 44a
    # The value 0.092 is 0.476 - 0.64 * 0.6 and is taken to limit the decrease but this equation is not valid in this
    # range anyway

    return np.array([anisotropy_q, anisotropy_q, (1.0 - 2.0 * anisotropy_q)])
