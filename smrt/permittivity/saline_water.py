import scipy as sc

from ..core.layer import required_layer_properties


@required_layer_properties("temperature", "salinity")
def seawater_permittivity_klein76(frequency, temperature, salinity):
    """Calculates permittivity (dielectric constant) of water using an empirical relationship described
       by Klein and Swift (1976).
       :param frequency: frequency in Hz
       :param temperature: water temperature in K
       :param salinity: water salinity in kg/kg (see PSU constant in smrt module)
       Returns complex water permittivity for a frequency f."""

    T = temperature - 273.15
    S = salinity*1e3
    f = frequency

    omega = 2 * sc.pi * f
    eps_0 = 8.854e-12  # permittivity of free space
    eps_inf = 4.9  # limiting high frequency value

    # calculate eps_s = static dielectric constant of saline water:
    eps_s_T = 87.134 - 1.949e-1 * T - 1.276e-2 * T**2 + 2.491e-4 * T**3
    a_ST = 1. + 1.613e-5 * S * T - 3.656e-3 * S + 3.210e-5 * S**2 - 4.232e-7 * S**3
    eps_s = eps_s_T * a_ST

    # calculate tau = relaxation time of saline water:
    tau_T0 = 1.768e-11 - 6.086e-13 * T + 1.104e-14 * T**2 - 8.111e-17 * T**3
    b_ST = 1. + 2.282e-5 * S * T - 7.638e-4 * S - 7.760e-6 * S**2 + 1.105e-8 * S**3
    tau = tau_T0 * b_ST

    # calculate sigma = ionic conductivity of dissolved salts:
    delta = 25 - T
    beta = 2.0333e-2 + 1.266e-4 * delta + 2.464e-6 * delta**2 - S * \
      (1.849e-5 - 2.551e-7 * delta + 2.551e-8 * delta**2)
    sigma_25S = S * (0.182521 - 1.46192e-3 * S + 2.09324e-5 * S**2 - 1.28205e-7 * S**3)
    sigma = sigma_25S * sc.exp(-delta * beta)

    # Debye type relaxation equation:
    eps_water = eps_inf + (eps_s - eps_inf) / (1 - 1j *
                                               omega * tau) + 1j * sigma / (omega * eps_0)
    return eps_water
