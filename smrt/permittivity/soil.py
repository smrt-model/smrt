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
def soil_permittivity_dobson85_peplinski95(frequency, temperature, moisture, sand, clay):
    """Compute the soil dielectric constant using the Dobson et al. (1985) formulation adapted by Peplinski et al., (1995).

    History:
        - equation implemented by M. Sandells (~2016)
        - coefficients check by Marion Leduc-Leballeur and M. Brogioni (2026)
        - added references and equation number by M. Leduc-Leballeur and G. Picard (2026)

    References:

    - Dobson, M. C., Ulaby, F. T., Hallikainen, M. T., & El-Rayes, M. A. (1985).
        Microwave dielectric behavior of wet soil—Part II: Dielectric mixing models.
        IEEE Transactions on Geoscience and Remote Sensing, GE-23(1), 35–46.

    - N. R. Peplinski, F. T. Ulaby and M. C. Dobson, "Dielectric properties of soils in the 0.3-1.3-GHz range," in IEEE
        Transactions on Geoscience and Remote Sensing, vol. 33, no. 3, pp. 803-807, May 1995, doi: 10.1109/36.387598.

    - A. Stogryn, "Equations for calculating the dielectric constant of saline water," IEEE Trans. Microwave Theory
        Tech., vol. MTT-19, pp. 733-736, 1971.
    """

    e_0 = PERMITTIVITY_OF_FREE_SPACE
    e_w_inf = 4.9
    e_s = 4.7
    rho_b = 1.3
    rho_s = 2.664

    temp = temperature - 273.15

    beta_prime = 1.2748 - 0.519 * sand - 0.152 * clay  # DB85 eq 30
    beta_second = 1.33797 - 0.603 * sand - 0.166 * clay  # DB85 eq 31

    sigma_eff = 0.0467 + 0.2204 * rho_b - 0.4111 * sand + 0.6614 * clay  # # eq 10 P95 (refitted based on eq 32 DB85)

    # static water permittivity referenced to Stogryn 1971 in DB95
    e_w0 = 87.134 - 1.949e-1 * temp - 1.276e-2 * temp**2 + 2.491e-4 * temp**3
    # relaxation time of water referenced to Stogryn 1971 in DB95
    rt_w = (1.1109e-10 - 3.824e-12 * temp + 6.938e-14 * temp**2 - 5.096e-16 * temp**3) / (2 * np.pi)

    e_fw_prime = e_w_inf + (e_w0 - e_w_inf) / (1 + (2 * np.pi * frequency * rt_w) ** 2)  # eq 6 P95  or eq 23 DB85
    e_fw_second = 2 * np.pi * frequency * rt_w * (e_w0 - e_w_inf) / (
        1 + (2 * np.pi * frequency * rt_w) ** 2
    ) + sigma_eff * (rho_s - rho_b) / (2 * np.pi * frequency * e_0 * rho_s * moisture)  # eq 7 P95 and eq 24 DB85

    return complex(
        (1 + (rho_b / rho_s) * (e_s**0.65 - 1) + moisture**beta_prime * e_fw_prime**0.65 - moisture)
        ** (1 / 0.65),  # eq 2 in P95 corrected with the missin -1 or Real(eq 28 in DB85)
        (moisture**beta_second * e_fw_second**0.65) ** (1 / 0.65),  # eq 3 in P95 or Imag(eq 28 in DB85)
    )


@layer_properties("temperature", "moisture", "sand", "clay")
def soil_permittivity_dobson85(frequency, temperature, moisture, sand, clay):
    """Compute the soil dielectric constant using the Dobson et al., (1985) formulation (original).

    It is not recommended to use this function, please use soil_permittivity_dobson85_peplinski95 instead.

    History:
        - added by M. Leduc-Leballeur and G. Picard (2026)

    References:

    - Dobson, M. C., Ulaby, F. T., Hallikainen, M. T., & El-Rayes, M. A. (1985).
        Microwave dielectric behavior of wet soil—Part II: Dielectric mixing models.
        IEEE Transactions on Geoscience and Remote Sensing, GE-23(1), 35–46.

    - N. R. Peplinski, F. T. Ulaby and M. C. Dobson, "Dielectric properties of soils in the 0.3-1.3-GHz range," in IEEE
        Transactions on Geoscience and Remote Sensing, vol. 33, no. 3, pp. 803-807, May 1995, doi: 10.1109/36.387598.

    - A. Stogryn, "Equations for calculating the dielectric constant of saline water," IEEE Trans. Microwave Theory
        Tech., vol. MTT-19, pp. 733-736, 1971.
    """

    e_0 = PERMITTIVITY_OF_FREE_SPACE
    e_w_inf = 4.9
    e_s = 4.7
    rho_b = 1.3
    rho_s = 2.664

    temp = temperature - 273.15

    beta_prime = 1.2748 - 0.519 * sand - 0.152 * clay  # DB85 eq 30
    beta_second = 1.33797 - 0.603 * sand - 0.166 * clay  # DB85 eq 31

    # original eq 32 DB85. Not used here because S has a different unit (permil->fraction) and interpretation (?) in DB85
    # sigma_eff = -1.645 + 1.939 * rho_b - 0.02013 * sand + 0.01594 * clay  #
    # equation eq 8 given in Peplinski et al., 1995
    # See also Ulaby 2014, section 4.8.1, page 252
    sigma_eff = -1.645 + 1.939 * rho_b - 2.25622 * sand + 1.594 * clay  # eq 8 P95

    # static water permittivity referenced to Stogryn 1971 in DB95
    e_w0 = 87.134 - 1.949e-1 * temp - 1.276e-2 * temp**2 + 2.491e-4 * temp**3
    # relaxation time of water referenced to Stogryn 1971 in DB95
    rt_w = (1.1109e-10 - 3.824e-12 * temp + 6.938e-14 * temp**2 - 5.096e-16 * temp**3) / (2 * np.pi)

    e_fw_prime = e_w_inf + (e_w0 - e_w_inf) / (1 + (2 * np.pi * frequency * rt_w) ** 2)  # eq 6 P95  or eq 23 DB85
    e_fw_second = 2 * np.pi * frequency * rt_w * (e_w0 - e_w_inf) / (
        1 + (2 * np.pi * frequency * rt_w) ** 2
    ) + sigma_eff * (rho_s - rho_b) / (2 * np.pi * frequency * e_0 * rho_s * moisture)  # eq 7 P95 and eq 24 DB85

    return complex(
        (1 + (rho_b / rho_s) * (e_s**0.65 - 1) + moisture**beta_prime * e_fw_prime**0.65 - moisture)
        ** (1 / 0.65),  # eq 2 in P95 corrected with the missin -1 or Real(eq 28 in DB85)
        (moisture**beta_second * e_fw_second**0.65) ** (1 / 0.65),  # eq 3 in P95 or Imag(eq 28 in DB85)
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
