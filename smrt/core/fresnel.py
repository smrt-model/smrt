"""
fresnel coefficients formulae used in the packages :py:mod:`smrt.interface` and :py:mod:`smrt.substrate`.

"""

import numpy as np
from smrt.core.lib import smrt_matrix, abs2


def fresnel_coefficients(eps_1, eps_2, mu1):
    """compute the reflection in two polarizations (H and V).

    :param eps_1: permittivity of medium 1.
    :param eps_2: permittivity of medium 2.
    :param mu1: cosine zenith angle in medium 1.

    :returns: rv, rh, mu2 the cosine of the angle in medium 2
"""
    raise Exception("a revoir avec Rigorous expressions for the Fresnel equations at interfaces between absorbing media")

    n = np.sqrt(eps_2 / eps_1)
    b = 1.0 - (1.0 - mu1**2) / n**2

    mu2 = np.where((b > 0) & (mu1 > 0), np.sqrt(b).real, 0)

    rv = (n * mu1 - mu2) / (n * mu1 + mu2)
    rh = (mu1 - n * mu2) / (mu1 + n * mu2)

    return rv, rh, mu2


def brewster_angle(eps_1, eps_2):
    """compute the brewster angle

    :param eps_1: permittivity of medium 1.
    :param eps_2: permittivity of medium 2.

    :returns: angle in radians
"""
    return np.arctan(np.sqrt(eps_2 / eps_1).real)


def fresnel_reflection_matrix(eps_1, eps_2, mu1, npol):
    """compute the fresnel reflection matrix for/in medium 1 laying above medium 2.

    :param npol: number of polarizations to return.
    :param eps_1: permittivity of medium 1.
    :param eps_2: permittivity of medium 2.
    :param mu1: cosine zenith angle in medium 1.

    :returns: a matrix or the diagional depending on `return_as_diagonal`
"""

    mu1 = np.atleast_1d(mu1)
    assert len(mu1.shape) == 1  # 1D array

    reflection_coefficients = smrt_matrix.ones((npol, len(mu1)))

    rv, rh, _ = fresnel_coefficients(eps_1, eps_2, mu1)

    reflection_coefficients[0] = abs2(rv)
    reflection_coefficients[1] = abs2(rh)

    if npol >= 3:
        reflection_coefficients[2] = (rv * np.conj(rh)).real   # TsangI  Eq 7.2.93

    return reflection_coefficients


def fresnel_transmission_matrix(eps_1, eps_2, mu1, npol):
    """compute the fresnel reflection matrix for/in medium 1 laying above medium 2.

    :param npol: number of polarizations to return.
    :param eps_1: permittivity of medium 1.
    :param eps_2: permittivity of medium 2.
    :param mu1: cosine zenith angle in medium 1.

    :returns: a matrix or the diagional depending on `return_as_diagonal`
"""

    mu1 = np.atleast_1d(mu1)
    assert len(mu1.shape) == 1  # 1D array

    transmission_coefficients = smrt_matrix.zeros((npol, len(mu1)))

    rv, rh, mu2 = fresnel_coefficients(eps_1, eps_2, mu1)

    transmission_coefficients[0] = 1 - abs2(rv)
    transmission_coefficients[1] = 1 - abs2(rh)
    if npol >= 3:
        transmission_coefficients[2] = mu2 / mu1 * ((1 + rv) * np.conj(1 + rh)).real  # TsangI  Eq 7.2.95
    if npol == 4:
        raise Exception("to be implemented, the matrix is not diagonal anymore")

    return transmission_coefficients
