"""
fresnel coefficients formulae used in the packages :py:mod:`smrt.interface` and :py:mod:`smrt.substrate`.

"""

import numpy as np
import scipy.sparse


def fresnel_coefficients(eps_1, eps_2, mu1):
    """compute the reflection in two polarizations (H and V)

    :param eps_1: permittivity of medium 1
    :param eps_2: permittivity of medium 2
    :param mu1: cosine zenith angle in medium 1

    :returns: rv, rh and mu2 the cosine of the angle in medium 2
"""
    n = np.sqrt(eps_2/eps_1)
    b = 1.0-(1.0-mu1**2)/(n**2)
    mask = b > 0

    mu2 = np.sqrt(b[mask]).real
    mu1_masked = mu1[mask]

    rv = (n*mu1_masked-mu2)/(n*mu1_masked+mu2)
    rh = (mu1_masked-n*mu2)/(mu1_masked+n*mu2)

    return rv, rh, mu2


def fresnel_reflection_matrix(eps_1, eps_2, mu1, npol, return_as_diagonal=False):
    """compute the fresnel reflection matrix for/in medium 1 laying above medium 2

    :param npol: number of polarizations to return
    :param eps_1: permittivity of medium 1
    :param eps_2: permittivity of medium 2
    :param mu1: cosine zenith angle in medium 1

    :returns: a matrix or the diagional depending on `return_as_diagonal`
"""

    mu1 = np.atleast_1d(mu1)
    assert len(mu1.shape) == 1  # 1D array

    reflection_coefficients = np.ones(npol*len(mu1))

    rv, rh, mu2 = fresnel_coefficients(eps_1, eps_2, mu1)

    neff = len(rv)  # number of reflection coefficient without total reflection

    reflection_coefficients[0::npol][0:neff] = np.abs(rv)**2
    reflection_coefficients[1::npol][0:neff] = np.abs(rh)**2

    if npol >= 3:
        reflection_coefficients[2::npol][0:neff] = (rv*np.conj(rh)).real   # TsangI  Eq 7.2.93
    if npol == 4:
        raise Exception("to be implemented, the matrix is not diagonal anymore")
    #reflection_coefficients[1::npol][mask] = reflection_coefficients[0::npol][mask] # test!!

    if return_as_diagonal:
        return reflection_coefficients
    else:
        return scipy.sparse.diags(reflection_coefficients, 0)  # create a diagonal matrix (reflection is only in the specular direction)


def fresnel_transmission_matrix(eps_1, eps_2, mu1, npol, return_as_diagonal=False):
    """compute the fresnel reflection matrix for/in medium 1 laying above medium 2

    :param npol: number of polarizations to return
    :param eps_1: permittivity of medium 1
    :param eps_2: permittivity of medium 2
    :param mu1: cosine zenith angle in medium 1

    :returns: a matrix or the diagional depending on `return_as_diagonal`
"""

    mu1 = np.atleast_1d(mu1)
    assert len(mu1.shape) == 1  # 1D array

    transmission_coefficients = np.zeros(npol*len(mu1))

    rv, rh, mu2 = fresnel_coefficients(eps_1, eps_2, mu1)

    neff = len(rv)  # number of reflection coefficient without total reflection

    transmission_coefficients[0::npol][0:neff] = 1 - np.abs(rv)**2
    transmission_coefficients[1::npol][0:neff] = 1 - np.abs(rh)**2
    if npol >= 3:
        transmission_coefficients[2:npol*neff:npol] = mu2 / mu1[0:neff] * ((1+rv)*np.conj(1+rh)).real  # TsangI  Eq 7.2.95
    if npol == 4:
        raise Exception("to be implemented, the matrix is not diagonal anymore")
    #transmission_coefficients[1::npol][mask] = transmission_coefficients[0::npol][mask] # test!!

    if return_as_diagonal:
        return transmission_coefficients
    else:
        return scipy.sparse.diags(transmission_coefficients, 0)  # create a diagonal matrix (reflection is only in the specular direction)
