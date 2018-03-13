import scipy as sc

from ..core.layer import required_layer_properties


@required_layer_properties("temperature", "salinity")
def seawater_permittivity_klein76(frequency, temperature, salinity):
    """Calculates permittivity (dielectric constant) of water using an empirical relationship described
       by Klein and Swift (1976).
       :param frequency: frequency in Hz
       :param temperature: water temperature in K
       :param salinity: water salinity [no units]
       Returns complex water permittivity for a frequency f."""

    T = temperature - 273.15
    S = salinity
    f = frequency

    omega = 2 * sc.pi * f
    eps_0 = 8.854 * 10**(-12)  # permittivity of free space
    eps_inf = 4.9  # limiting high frequency value

    # calculate eps_s = static dielectric constant of saline water:
    eps_s_T = 87.134 - 1.949 * \
        10**(-1) * T - 1.276 * 10**(-2) * T**2 + 2.491 * 10**(-4) * T**3
    a_ST = 1. + 1.613 * 10**(-5) * S * T - 3.656 * 10**(-3) * \
        S + 3.210 * 10**(-5) * S**2 - 4.232 * 10**(-7) * S**3
    eps_s = eps_s_T * a_ST

    # calculate tau = relaxation time of saline water:
    tau_T0 = 1.768 * 10**(-11) - 6.086 * 10**(-13) * T + \
        1.104 * 10**(-14) * T**2 - 8.111 * 10**(-17) * T**3
    b_ST = 1. + 2.282 * 10**(-5) * S * T - 7.638 * 10**(-4) * \
        S - 7.760 * 10**(-6) * S**2 + 1.105 * 10**(-8) * S**3
    tau = tau_T0 * b_ST

    # calculate sigma = ionic conductivity of dissolved salts:
    delta = 25 - T
    beta = 2.0333 * 10**(-2) + 1.266 * 10**(-4) * delta + 2.464 * 10**(-6) * delta**2 - \
        S * (1.849 * 10**(-5) - 2.551 * 10**(-7)
             * delta + 2.551 * 10**(-8) * delta**2)
    sigma_25S = S * (0.182521 - 1.46192 * 10**(-3) * S +
                     2.09324 * 10**(-5) * S**2 - 1.28205 * 10**(-7) * S**3)
    sigma = sigma_25S * sc.exp(-delta * beta)

    # Debye type relaxation equation:
    eps_water = eps_inf + (eps_s - eps_inf) / (1 - 1j *
                                               omega * tau) + 1j * sigma / (omega * eps_0)
    return eps_water
