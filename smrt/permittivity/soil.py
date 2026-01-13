"""This module implements various soil dielectric constant models.

References:

    - Dobson, M. C., Ulaby, F. T., Hallikainen, M. T., & El-Rayes, M. A. (1985).
        Microwave dielectric behavior of wet soil—Part II: Dielectric mixing models.
        IEEE Transactions on Geoscience and Remote Sensing, GE-23(1), 35–46.

    - Ulaby, F. T., Moore, R. K., & Fung, A. K. (1986).
        Microwave Remote Sensing: Active and Passive (Vol. III: From Theory to Applications).
        Artech House.

    - Pulliainen, J. T., Grandell, J., & Hallikainen, M. T. (1999). HUT snow emission model and its applicability
        to snow water equivalent retrieval. Geoscience and Remote Sensing, IEEE Transactions On, 37, 1378–1390.
        https://doi.org/10.1109/36.763302

    - Lemmetyinen, J., Pulliainen, J., Rees, A., Kontu, A., Qiu, Y., & Derksen, C. (2010). Multiple-Layer Adaptation
        of HUT Snow Emission Model: Comparison With Experimental Data. IEEE Transactions on Geoscience and Remote
        Sensing, 48, 2781–2794. https://doi.org/10.1109/TGRS.2010.2041357

    - Montpetit, B., Royer, A., Roy, A., & Langlois, A. (2018).
        In-situ passive microwave emission model parameterization of sub-arctic frozen organic soils.
        Remote Sensing of Environment, 205, 112–118. https://doi.org/10.1016/j.rse.2017.10.033


"""

import numpy as np
import scipy.interpolate

from smrt.core.error import SMRTError
from smrt.core.globalconstants import PERMITTIVITY_OF_FREE_SPACE
from smrt.core.layer import layer_properties


@layer_properties("temperature", "moisture", "sand", "clay")
def soil_permittivity_dobson85(frequency, temperature, moisture, sand, clay):
    """Compute the soil dielectric constant using the Dobson et al., (1985) formulation."""

    e_0 = PERMITTIVITY_OF_FREE_SPACE
    e_w_inf = 4.9
    e_s = 4.7
    rho_b = 1.3
    rho_s = 2.664

    temp = temperature - 273.15

    beta1 = 1.2748 - 0.519 * sand - 0.152 * clay
    beta2 = 1.33797 - 0.603 * sand - 0.166 * clay

    sigma_eff = 0.0467 + 0.2204 * rho_b - 0.4111 * sand + 0.6614 * clay

    e_w0 = 87.134 - 1.949e-1 * temp - 1.276e-2 * temp**2 + 2.491e-4 * temp**3
    rt_w = (1.1109e-10 - 3.824e-12 * temp + 6.938e-14 * temp**2 - 5.096e-16 * temp**3) / (2 * np.pi)

    e_fw1 = e_w_inf + (e_w0 - e_w_inf) / (1 + (2 * np.pi * frequency * rt_w) ** 2)
    e_fw2 = 2 * np.pi * frequency * rt_w * (e_w0 - e_w_inf) / (1 + (2 * np.pi * frequency * rt_w) ** 2) + sigma_eff * (
        rho_s - rho_b
    ) / (2 * np.pi * frequency * e_0 * rho_s * moisture)

    return complex(
        (1 + (rho_b / rho_s) * (e_s**0.65 - 1) + moisture**beta1 * e_fw1**0.65 - moisture) ** (1 / 0.65),
        (moisture**beta2 * e_fw2**0.65) ** (1 / 0.65),
    )


@layer_properties("temperature", "moisture", "sand", "clay", "dry_matter")
def soil_permittivity_hut(frequency, temperature, moisture, sand, clay, dry_matter):
    """Compute the soil dielectric constant using the HUT (Pulliainen et al., 1999; Lemmetyinen et al., 2010) formulation."""
    # Parameters for soil dielectric constant calculation with water
    ew_inf = 4.9

    tempC = temperature - 273.15

    if tempC >= 0:  # liquid water
        # calculates real and imag. part of water dielectricity (code HUT 20.12.95 [epsw.m]; K.Tigerstedt)
        ew0 = 87.74 - 0.40008 * tempC + 9.398e-4 * tempC**2 + 1.410e-6 * tempC**3
        # d = 25 - tempC # unused
        # alfa = 2.033e-2 + 1.266e-4 * d + 2.464e-6 * d**2 # unused
        tw = 1 / (2 * np.pi) * (1.1109e-10 - 3.824e-12 * tempC + 6.938e-14 * tempC**2 - 5.096e-16 * tempC**3)

        ew_r = ew_inf + (ew0 - ew_inf) / (1 + (2 * np.pi * frequency * tw) ** 2)
        ew_i = (ew0 - ew_inf) * 2 * np.pi * frequency * tw / (1 + (2 * np.pi * frequency * tw) ** 2)
    else:
        raise SMRTError("soil_permittivity_hut requires above freezing point temperatures")
    #      !option for salt consideration (Mätzler 1987)
    #      !iei_S =A/M+B*M**C                 !impure ice
    #      !iei_P=Ap/M+Bp*M**Cp                 !pure ice
    #      !delta_iei = iei_S - iei_P
    #      !ew_i=ew_i+delta_iei*SS/13

    beta = 1.09 - 0.11 * sand + 0.18 * clay
    # dm_rho is now in SI // Ulaby et al. (1986, p. 2099)
    epsalf = 1 + 0.65 * dry_matter / 1000.0 + moisture**beta * (complex(ew_r, ew_i) ** 0.65 - 1)

    return (epsalf) ** (1 / 0.65)


@layer_properties("temperature")
def soil_permittivity_montpetit08(frequency, temperature):
    """
    Computes the soil dielectric constant using the Montpetit et al. (2018) formulation.

    The formulation is only valid for below-freezing point temperature.

    Reference:
        Montpetit, B., Royer, A., Roy, A., & Langlois, A. (2018). In-situ passive microwave emission model
        parameterization of sub-arctic frozen organic soils. Remote Sensing of Environment, 205, 112–118.
        https://doi.org/10.1016/j.rse.2017.10.033

    Args:
        frequency: Frequency in Hz.
        temperature: Temperature in Kelvin.

    Returns:
        complex: Soil dielectric constant.
    """
    # from functools import partial
    # from smrt.inputs.make_soil import soil_permittivity_dobson
    if temperature > 273.15:
        raise SMRTError("soil_permittivity_monpetit is not implemented for above freezing temperatures.")
        # moisture=0.2
        # sand=0.4
        # clay=0.3
        # return partial(soil_permittivity_dobson, SM=moisture, S=sand, C=clay)
    # else:

    p = scipy.interpolate.interp1d(
        [10.65e9, 19e9, 37e9],
        [complex(3.18, 0.0061), complex(3.42, 0.0051), complex(4.47, 0.33)],
        fill_value="extrapolate",
    )
    return p(frequency)
