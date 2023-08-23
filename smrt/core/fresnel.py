"""
fresnel coefficients formulae used in the packages :py:mod:`smrt.interface` and :py:mod:`smrt.substrate`.

"""

import numpy as np
from smrt.core.lib import smrt_matrix, abs2


def fresnel_coefficients_old(eps_1, eps_2, mu1):
    """compute the reflection in two polarizations (H and V). The equations are only valid for lossless media.
    Applying these equations for (strongly) lossy media result in (large) errors. Don't use it. It is here for reference only.

    :param eps_1: permittivity of medium 1.
    :param eps_2: permittivity of medium 2.
    :param mu1: cosine zenith angle in medium 1.

    :returns: rv, rh, mu2 the cosine of the angle in medium 2
"""
    n = np.sqrt(eps_2 / eps_1)
    b = 1.0 - (1.0 - mu1**2) / n**2

    mu2 = np.where((b > 0) & (mu1 > 0), np.sqrt(b).real, 0)

    rv = (n * mu1 - mu2) / (n * mu1 + mu2)
    rh = (mu1 - n * mu2) / (mu1 + n * mu2)

    return rv, rh, mu2


def fresnel_coefficients_maezawa09_classical(eps_1, eps_2, mu1, full_output=False):
    """compute the reflection in two polarizations (H and V) for lossly media with the "classical Fresnel" based
    on Maezawa, H., & Miyauchi, H. (2009). Rigorous expressions for the Fresnel equations at interfaces between absorbing media. 
    Journal of the Optical Society of America A, 26(2), 330. https://doi.org/10.1364/josaa.26.000330

    The classical derivation does not respect energy conservation, especially the transmittivity.
    Don't use it. It is here for reference only.

    :param eps_1: permittivity of medium 1.
    :param eps_2: permittivity of medium 2.
    :param mu1: cosine zenith angle in medium 1.

    :returns: rv, rh, mu2 the cosine of the angle in medium 2
"""
    # y is the axis normal to the interface (usually z, but here it is y!)

    # incident wavenumber
    n1 = np.sqrt(eps_1)

    kiz2 = n1.real**2 * (1 - mu1**2)   # kiz = n1 * sin(theta)
    kyi = - np.sqrt(eps_1 - kiz2)                  # Eq 8 for i

    ktz2 = kiz2   # unnumbered equation before 22  -> tangential k is conserved throught the interface (=Snell law)
    kyt = - np.sqrt(complex(eps_2) - ktz2)                  # Eq 8 for t

    rh = (kyi - kyt) / (kyi + kyt)  # Eq 30

    rv = (eps_2 * kyi - eps_1 * kyt) / (eps_2 * kyi + eps_1 * kyt)  # Eq 32

    mu2 = - kyt.real / np.sqrt(eps_2).real  # by definition of kyt

    if full_output:
        n2 = np.sqrt(eps_2)

        th = 2 * kyi / (kyi + kyt)  # Eq 31

        tv = 2 * n1 * n2 * kyi / (eps_2 * kyi + eps_1 * kyt)   # Eq 33
        Rv = abs2(rv)  # Eq 34
        Rh = abs2(rh)  # Eq 34
        # Th = (kyt + kyt.conjugate()) / (kyi + kyi.conjugate()) * abs2(th)  # Eq 35
        Th = kyt.real / kyi.real * abs2(th)  # Optimized Eq 35

        # Tv = abs2(n1) * (eps_2.conjugate() * kyt + eps_2 * kyt.conjugate()) \
        #     / (abs2(n2) * (eps_1.conjugate() * kyi + eps_1 * kyi.conjugate())) \
        #     * abs2(tv)                                                      # Eq 36
        Tv = abs2(n1) * (eps_2.conjugate() * kyt).real \
            / (abs2(n2) * (eps_1.conjugate() * kyi).real) \
            * abs2(tv)                                                      # Optimized Eq 36

        return rv, rh, th, tv, Rv, Rh, Tv, Th, mu2
    else:
        return rv, rh, mu2


def fresnel_coefficients_maezawa09_rigorous(eps_1, eps_2, mu1, full_output=False):
    """compute the reflection in two polarizations (H and V) for lossly media with the "rigorous Fresnel" based
    on Maezawa, H., & Miyauchi, H. (2009). Rigorous expressions for the Fresnel equations at interfaces between absorbing media. 
    Journal of the Optical Society of America A, 26(2), 330. https://doi.org/10.1364/josaa.26.000330

    The 'rigorous' derivation respect the energy conservation even for strongly loosly media.

    :param eps_1: permittivity of medium 1.
    :param eps_2: permittivity of medium 2.
    :param mu1: cosine zenith angle in medium 1.

    :returns: rv, rh, mu2 the cosine of the angle in medium 2
"""
    # y is the axis normal to the interface (usually z, but here it is y!)

    # incident wavenumber
    n1 = np.sqrt(eps_1)
    kiz2 = n1.real**2 * (1 - mu1**2)   # this the square of kiz = n1 * sin(theta)
    kyi = - np.sqrt(eps_1 - kiz2)                  # Eq 8 for i

    ktz2 = kiz2   # unnumbered equation before 22  -> tangential k is conserved throught the interface (=Snell law)
    kyt = - np.sqrt(complex(eps_2) - ktz2)                  # Eq 8 for t

    rh = (kyi - kyt) / (kyi.conjugate() + kyt)  # Eq 59

    rv = n1.conjugate() * (eps_2 * kyi - eps_1 * kyt) / (n1 * (eps_2 * kyi.conjugate() + eps_1.conjugate() * kyt))  # Eq 61

    mu2 = - kyt.real / np.sqrt(eps_2).real  # by definition of kyt

    if full_output:
        n2 = np.sqrt(eps_2)

        th = 2 * kyi.real / (kyi.conjugate() + kyt)  # Eq 60

        tv = n2 * 2 * (eps_1.conjugate() * kyi).real / (n1 * (eps_2 * kyi.conjugate() + eps_1.conjugate() * kyt))   # Eq 62

        Rv = abs2(rv)  # Eq 34
        Rh = abs2(rh)  # Eq 34
        # Th = (kyt + kyt.conjugate()) / (kyi + kyi.conjugate()) * abs2(th)  # Eq 35
        Th = kyt.real / kyi.real * abs2(th)  # Optimized Eq 35

        # Tv = abs2(n1) * (eps_2.conjugate() * kyt + eps_2 * kyt.conjugate()) \
        #     / (abs2(n2) * (eps_1.conjugate() * kyi + eps_1 * kyi.conjugate())) \
        #     * abs2(tv)                                                      # Eq 36
        Tv = abs2(n1) * (eps_2.conjugate() * kyt).real \
            / (abs2(n2) * (eps_1.conjugate() * kyi).real) \
            * abs2(tv)                                                      # Optimized Eq 36

        assert np.allclose(Rv + Tv, 1)  # check energy conservation
        assert np.allclose(Rh + Th, 1)  # check energy conservation

        return rv, rh, th, tv, Rv, Rh, Tv, Th, mu2
    else:
        return rv, rh, mu2


# use the best function for the fresnel coefficients
fresnel_coefficients = fresnel_coefficients_maezawa09_rigorous


def snell_angle(eps_1, eps_2, mu1):
    """compute mu2 the cos(angle) in the second medium according to Snell's law."""

    # incident wavenumber
    n1 = np.sqrt(eps_1)
    kiz2 = n1.real**2 * (1 - mu1**2)   # this the square of kiz = n1 * sin(theta)

    ktz2 = kiz2   # unnumbered equation before 22  -> tangential k is conserved throught the interface (=Snell law)
    kyt = - np.sqrt(complex(eps_2) - ktz2)                  # Eq 8 for t

    mu2 = - kyt.real / np.sqrt(eps_2).real  # by definition of kyt

    return mu2


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
        # It is not sure this equation is valid for strongly loosly materails

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
        # It is not sure this equation is valid for strongly loosly materails

    if npol == 4:
        raise Exception("to be implemented, the matrix is not diagonal anymore")

    return transmission_coefficients
