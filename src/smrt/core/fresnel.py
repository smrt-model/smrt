"""Fresnel coefficients formulae used in the packages :py:mod:`smrt.interface` and :py:mod:`smrt.substrate`."""

import numba
import numpy as np

from smrt.core.error import SMRTError
from smrt.core.lib import abs2, smrt_matrix


def fresnel_reflection_coefficients_old(eps_1, eps_2, mu1):
    """Compute the reflection in two polarizations (H and V). The equations are only valid for lossless media. Applying
    these equations for (strongly) lossy media results in (large) errors. Don't use it. It is here for reference only.
    The returned reflection coefficients apply to the electric field. Use abs2(rv), abs2(rh) to obtain the power
    reflection coefficient.

    Args:
        eps_1: permittivity of medium 1.
        eps_2: permittivity of medium 2.
        mu1: cosine zenith angle in medium 1.

    Returns:
        rv: complex
            reflection coefficient for vertical polarization.
        rh: complex
            reflection coefficient for horizontal polarization.
        mu2: float
            cosine of the angle in medium 2.
    """
    n = np.sqrt(eps_2 / eps_1)
    b = 1.0 - (1.0 - mu1**2) / n**2

    mu2 = np.where((b > 0) & (mu1 > 0), np.sqrt(b).real, 0)

    rv = (n * mu1 - mu2) / (n * mu1 + mu2)
    rh = (mu1 - n * mu2) / (mu1 + n * mu2)

    return rv, rh, mu2


def fresnel_coefficients_maezawa09_classical(eps_1, eps_2, mu, mu_medium="1", full_output=False):
    """Compute the reflection in two polarizations (H and V) for lossy media with the "classical Fresnel" based
    on Maezawa, H., & Miyauchi, H. (2009). Rigorous expressions for the Fresnel equations at interfaces between
    absorbing media. Journal of the Optical Society of America A, 26(2), 330. https://doi.org/10.1364/josaa.26.000330

    The classical derivation does not respect energy conservation, especially the transmittivity.
    Don't use it. It is here for reference only.
    The returned reflection coefficients apply to the electric field. Use abs2(rv), abs2(rh) to obtain the power
    reflection coefficient.

    Args:
        eps_1: permittivity of medium 1.
        eps_2: permittivity of medium 2.
        mu: cosine zenith angle in medium 1 or in the void according to mu_medium.
        mu_medium: string indicating which medium the angle is defined in ("1" or "void").
        full_output: bool
            if True, return additional transmission and power coefficients.

    Returns:
        rv: complex
            reflection coefficient for vertical polarization.
        rh: complex
            reflection coefficient for horizontal polarization.
        mu2: float
            cosine of the angle in medium 2.
    """
    # y is the axis normal to the interface (usually z, but here it is y!)

    # incident wavenumber
    n1 = np.sqrt(eps_1)
    n2 = np.sqrt(eps_2)

    if mu_medium == "1":
        kiz2 = n1.real**2 * (1 - mu**2)  # this the square of kiz = n1 * sin(theta)
    elif mu_medium == "void":
        kiz2 = 1 - mu**2  # this the square of kiz = sin(theta0)
    else:
        raise SMRTError("mu_medium must be either '1' or 'void'")

    kyi = -np.sqrt(eps_1 - kiz2)  # Eq 8 for i

    ktz2 = kiz2  # unnumbered equation before 22  -> tangential k is conserved throught the interface (=Snell law)
    kyt = -np.sqrt(eps_2 - ktz2, dtype=np.complex128)  # Eq 8 for t

    rh = (kyi - kyt) / (kyi + kyt)  # Eq 30

    rv = (eps_2 * kyi - eps_1 * kyt) / (eps_2 * kyi + eps_1 * kyt)  # Eq 32

    mu2 = -kyt.real / n2.real  # by definition of kyt

    if full_output:
        th = 2 * kyi / (kyi + kyt)  # Eq 31

        tv = 2 * n1 * n2 * kyi / (eps_2 * kyi + eps_1 * kyt)  # Eq 33
        Rv = abs2(rv)  # Eq 34
        Rh = abs2(rh)  # Eq 34
        # Th = (kyt + kyt.conjugate()) / (kyi + kyi.conjugate()) * abs2(th)  # Eq 35
        Th = kyt.real / kyi.real * abs2(th)  # Optimized Eq 35

        # Tv = abs2(n1) * (eps_2.conjugate() * kyt + eps_2 * kyt.conjugate()) \
        #     / (abs2(n2) * (eps_1.conjugate() * kyi + eps_1 * kyi.conjugate())) \
        #     * abs2(tv)                                                      # Eq 36
        Tv = (
            abs2(n1) * (eps_2.conjugate() * kyt).real / (abs2(n2) * (eps_1.conjugate() * kyi).real) * abs2(tv)
        )  # Optimized Eq 36

        return rv, rh, tv, th, Rv, Rh, Tv, Th, mu2
    else:
        return rv, rh, mu2


def fresnel_reflection_coefficients_maezawa09_rigorous(
    eps_1, eps_2, mu1=None, *, mu0=None
) -> tuple[complex, complex, float]:
    """Compute the reflection in two polarizations (H and V) for lossy media with the "rigorous Fresnel" based
    on Maezawa, H., & Miyauchi, H. (2009). Rigorous expressions for the Fresnel equations at interfaces between
    absorbing media. Journal of the Optical Society of America A, 26(2), 330. https://doi.org/10.1364/josaa.26.000330

    The 'rigorous' derivation respects the energy conservation even for strongly lossy media.
    The returned reflection coefficients apply to the electric field. Use abs2(rv), abs2(rh) to obtain the power
    reflection coefficient.

    This function only returns the FIELD reflection coefficients and the cosine angle in the medium 2

    Args:
        eps_1: permittivity of medium 1.
        eps_2: permittivity of medium 2.
        mu: cosine zenith angle in medium 1 or in the void according to mu_medium.
        mu0: cosine zenith angle in the void (real, scalar or array). mu must be None if mu0 is provided.
    Returns:
        rv: complex
            reflection coefficient for vertical polarization.
        rh: complex
            reflection coefficient for horizontal polarization.
        mu2: float
            cosine of the angle in medium 2.
    """
    # y is the axis normal to the interface (usually z, but here it is y!)

    # incident wavenumber
    n1 = np.sqrt(eps_1)
    n2 = np.sqrt(eps_2)

    if mu1 is not None:
        assert mu0 is None
        kiz2 = n1.real**2 * (1 - mu1**2)  # this the square of kiz = n1 * sin(theta)
    else:
        kiz2 = 1 - mu0**2  # this the square of kiz = sin(theta0)

    kyi = -np.sqrt(eps_1 - kiz2, dtype=np.complex128)  # Eq 8 for i

    ktz2 = kiz2  # unnumbered equation before 22  -> tangential k is conserved throught the interface (=Snell law)
    kyt = -np.sqrt(eps_2 - ktz2, dtype=np.complex128)  # Eq 8 for t

    rh = (kyi - kyt) / (kyi.conjugate() + kyt)  # Eq 59

    rv = (
        n1.conjugate() * (eps_2 * kyi - eps_1 * kyt) / (n1 * (eps_2 * kyi.conjugate() + eps_1.conjugate() * kyt))
    )  # Eq 61
    # n1.conjugate() / n1 can be replaced by eps_1.conjugate / abs(eps_1)

    mu2 = np.sqrt(np.maximum(1 - kiz2 / n2.real**2, 0.0))

    return rv, rh, mu2


#
#
##### fresnel_reflection_coefficients_maezawa09_rigorous_compiled ####
#
#


@numba.jit(nopython=True, cache=True)
def fresnel_reflection_coefficients_maezawa09_rigorous_compiled(eps_1, eps_2, mu1):
    """Compute the reflection in two polarizations (H and V) for lossy media with the "rigorous Fresnel" based
    on Maezawa, H., & Miyauchi, H. (2009). Rigorous expressions for the Fresnel equations at interfaces between
    absorbing media. Journal of the Optical Society of America A, 26(2), 330. https://doi.org/10.1364/josaa.26.000330

    The 'rigorous' derivation respects the energy conservation even for strongly lossy media.
    The returned reflection coefficients apply to the electric field. Use abs2(rv), abs2(rh) to obtain the power
    reflection coefficient.

    This function only returns the FIELD reflection coefficients and the cosine angle in the medium 2

    Args:
        eps_1: permittivity of medium 1 (real or complex, scalar or array).
        eps_2: permittivity of medium 2 (real or complex, scalar or array).
        mu1: cosine zenith angle in medium 1 (real, scalar or array).

    Returns:
        rv, rh, mu2: field reflection coefficients (V, H) and cosine angle in medium 2, with the broadcast shape
        of the inputs.
    """
    eps_1 = eps_1 + 0j  # promote real inputs to complex
    eps_2 = eps_2 + 0j
    kiz2 = np.sqrt(eps_1).real ** 2 * (1 - mu1**2)  # square of n1 * sin(theta)

    return _fresnel_reflection_maezawa09_rigorous_reflection_core(eps_1, eps_2, kiz2)


@numba.jit(nopython=True, cache=True)
def fresnel_reflection_coefficients_maezawa09_rigorous_mu0_compiled(eps_1, eps_2, *, mu0):
    """Same as :py:func:`fresnel_reflection_coefficients_maezawa09_rigorous_compiled`
    with the angle defined in the void.

    Args:
        eps_1: permittivity of medium 1 (real or complex, scalar or array).
        eps_2: permittivity of medium 2 (real or complex, scalar or array).
        mu0: cosine zenith angle in the void (real, scalar or array).

    Returns:
        rv, rh, mu2: field reflection coefficients (V, H) and cosine angle in medium 2, with the broadcast shape
        of the inputs.
    """
    eps_1 = eps_1 + 0j  # promote real inputs to complex
    eps_2 = eps_2 + 0j
    kiz2 = 1 - mu0**2  # square of sin(theta0)

    return _fresnel_reflection_maezawa09_rigorous_reflection_core(eps_1, eps_2, kiz2)


@numba.jit(nopython=True, cache=True)
def _fresnel_reflection_maezawa09_rigorous_reflection_core(eps_1, eps_2, kiz2):
    # eps_1 and eps_2 are complex (scalar or array); kiz2 is the squared tangential wavenumber (real).
    n1 = np.sqrt(eps_1)
    n2 = np.sqrt(eps_2)

    kyi = -np.sqrt(eps_1 - kiz2)  # Eq 8 for i
    kyt = -np.sqrt(eps_2 - kiz2)  # Eq 8 for t (tangential k is conserved: Snell law)

    rh = (kyi - kyt) / (kyi.conjugate() + kyt)  # Eq 59
    rv = (
        n1.conjugate() * (eps_2 * kyi - eps_1 * kyt) / (n1 * (eps_2 * kyi.conjugate() + eps_1.conjugate() * kyt))
    )  # Eq 61

    mu2 = np.sqrt(np.maximum(1 - kiz2 / n2.real**2, 0.0))

    return rv, rh, mu2


#
#
##### fresnel_coefficients_maezawa09_rigorous_compiled ####
#
#


@numba.jit(nopython=True, cache=True)
def fresnel_coefficients_maezawa09_rigorous_compiled(eps_1, eps_2, mu1):
    """Compute the FIELD reflection and transmission coefficients in two polarizations (H and V) for lossy media with
    the "rigorous Fresnel" based on Maezawa, H., & Miyauchi, H. (2009). Rigorous expressions for the Fresnel equations
    at interfaces between absorbing media. Journal of the Optical Society of America A, 26(2), 330.
    https://doi.org/10.1364/josaa.26.000330

    The 'rigorous' derivation respects the energy conservation even for strongly lossy media.
    Use abs2(rv), abs2(rh) to obtain the power reflection coefficient. The angle is defined in medium 1; for the angle
    defined in the void, use :py:func:`fresnel_coefficients_maezawa09_rigorous_muvoid_compiled`.

    Args:
        eps_1: permittivity of medium 1 (real or complex, scalar or array).
        eps_2: permittivity of medium 2 (real or complex, scalar or array).
        mu1: cosine zenith angle in medium 1 (real, scalar or array).

    Returns:
        rv, rh, tv, th, mu2: field reflection and transmission coefficients (V, H) and cosine angle in medium 2,
        with the broadcast shape of the inputs.
    """
    eps_1 = eps_1 + 0j  # promote real inputs to complex
    eps_2 = eps_2 + 0j
    kiz2 = np.sqrt(eps_1).real ** 2 * (1 - mu1**2)  # square of n1 * sin(theta)

    return _fresnel_coefficients_maezawa09_rigorous_core(eps_1, eps_2, kiz2)


@numba.jit(nopython=True, cache=True)
def fresnel_coefficients_maezawa09_rigorous_mu0_compiled(eps_1, eps_2, *, mu0):
    """Same as :py:func:`fresnel_coefficients_maezawa09_rigorous_compiled` with the angle defined in the void.

    Args:
        eps_1: permittivity of medium 1 (real or complex, scalar or array).
        eps_2: permittivity of medium 2 (real or complex, scalar or array).
        mu0: cosine zenith angle in the void (real, scalar or array).

    Returns:
        rv, rh, tv, th, mu2: field reflection and transmission coefficients (V, H) and cosine angle in medium 2,
        with the broadcast shape of the inputs.
    """
    eps_1 = eps_1 + 0j  # promote real inputs to complex
    eps_2 = eps_2 + 0j
    kiz2 = 1 - mu0**2  # square of sin(theta0)

    return _fresnel_coefficients_maezawa09_rigorous_core(eps_1, eps_2, kiz2)


@numba.jit(nopython=True, cache=True)
def _fresnel_coefficients_maezawa09_rigorous_core(eps_1, eps_2, kiz2):
    # eps_1 and eps_2 are complex (scalar or array); kiz2 is the squared tangential wavenumber (real).
    n1 = np.sqrt(eps_1)
    n2 = np.sqrt(eps_2)

    # M09: We take the minus sign for the incidence and transmission field and plus sign for the reflection field.
    kyi = -np.sqrt(eps_1 - kiz2)  # Eq 8 for i
    kyt = -np.sqrt(eps_2 - kiz2)  # Eq 8 for t (tangential k is conserved: Snell law)

    denom_h = kyi.conjugate() + kyt
    rh = (kyi - kyt) / denom_h  # Eq 59
    th = 2 * kyi.real / denom_h  # Eq 60

    denom_v = n1 * (eps_2 * kyi.conjugate() + eps_1.conjugate() * kyt)
    rv = n1.conjugate() * (eps_2 * kyi - eps_1 * kyt) / denom_v  # Eq 61
    tv = n2 * 2 * (eps_1.conjugate() * kyi).real / denom_v  # Eq 62

    # the real part of n2 is used instead of the complex cosine (positive imag branch)
    mu2 = np.sqrt(np.maximum(1 - kiz2 / n2.real**2, 0.0))

    return rv, rh, tv, th, mu2


def fresnel_reflection_coefficients_maezawa09_rigorous_full_output(eps_1, eps_2, mu1=None, *, mu0=None):
    """Compute the reflection in two polarizations (H and V) for lossy media with the "rigorous Fresnel" based
    on Maezawa, H., & Miyauchi, H. (2009). Rigorous expressions for the Fresnel equations at interfaces between
    absorbing media. Journal of the Optical Society of America A, 26(2), 330. https://doi.org/10.1364/josaa.26.000330

    The 'rigorous' derivation respects the energy conservation even for strongly lossy media.
    The returned reflection coefficients apply to the electric field. Use abs2(rv), abs2(rh) to obtain the power
    reflection coefficient.

    This function returns the FIELD and INTENSITY reflection and transmission coefficients and the cosine angle in the
    medium 2

    Args:
        eps_1: permittivity of medium 1.
        eps_2: permittivity of medium 2.
        mu: cosine zenith angle in medium 1 or in the void according to mu_medium.
        mu_medium: string indicating which medium the angle is defined in ("1" or "void").

    Returns:
        rv: complex
            reflection coefficient for vertical polarization.
        rh: complex
            reflection coefficient for horizontal polarization.
        tv: complex
            transmission coefficient for vertical polarization.
        th: complex
            transmission coefficient for horizontal polarization.
        Rv: float
            power reflection coefficient for vertical polarization.
        Rh: float
            power reflection coefficient for horizontal polarization.
        Tv: float
            power transmission coefficient for vertical polarization.
        Th: float
            power transmission coefficient for horizontal polarization.
        mu2: float
            cosine of the angle in medium 2.
    """
    # y is the axis normal to the interface (usually z, but here it is y!)

    # this part is the same as in fresnel_reflection_coefficients_maezawa09_rigorous
    # the duplication is for numba compilation to work as returning different set of parameters is not allowed

    # incident wavenumber

    n1 = np.sqrt(eps_1)
    n2 = np.sqrt(eps_2)

    if mu1 is not None:
        assert mu0 is None
        kiz2 = n1.real**2 * (1 - mu1**2)  # this the square of kiz = n1 * sin(theta)
    elif mu0 is not None:
        kiz2 = 1 - mu0**2  # this the square of kiz = sin(theta0)
    else:
        raise SMRTError("mu_medium must be either '1' or 'void'")

    # M09: We take the minus sign for the incidence and transmission field and plus sign for the reflection field.
    kyi = -np.sqrt(eps_1 - kiz2, dtype=np.complex128)  # Eq 8 for i

    ktz2 = kiz2  # unnumbered equation before 22  -> tangential k is conserved throught the interface (=Snell law)
    kyt = -np.sqrt(eps_2 - ktz2, dtype=np.complex128)  # Eq 8 for t

    rh = (kyi - kyt) / (kyi.conjugate() + kyt)  # Eq 59

    rv = (
        n1.conjugate() * (eps_2 * kyi - eps_1 * kyt) / (n1 * (eps_2 * kyi.conjugate() + eps_1.conjugate() * kyt))
    )  # Eq 61

    # Snell's law
    mu2 = np.sqrt(np.maximum(1 - kiz2 / n2.real**2, 0.0))

    # this part is for the full_output
    # full_output
    #

    th = 2 * kyi.real / (kyi.conjugate() + kyt)  # Eq 60; in the paper is with kyi + kyi.conjugate() == 2 * kyi.real

    tv = n2 * 2 * (eps_1.conjugate() * kyi).real / (n1 * (eps_2 * kyi.conjugate() + eps_1.conjugate() * kyt))  # Eq 62

    Rv = abs2(rv)  # Eq 34
    Rh = abs2(rh)  # Eq 34
    # Th = (kyt + kyt.conjugate()) / (kyi + kyi.conjugate()) * abs2(th)  # Eq 35
    Th = kyt.real / kyi.real * abs2(th)  # Optimized Eq 35

    # Tv = abs2(n1) * (eps_2.conjugate() * kyt + eps_2 * kyt.conjugate()) \
    #     / (abs2(n2) * (eps_1.conjugate() * kyi + eps_1 * kyi.conjugate())) \
    #     * abs2(tv)                                                      # Eq 36
    Tv = (
        abs2(n1) * (eps_2.conjugate() * kyt).real / (abs2(n2) * (eps_1.conjugate() * kyi).real) * abs2(tv)
    )  # Optimized Eq 36

    assert np.allclose(Rv + Tv, 1), f"Energy conservation error in Rv + Tv: {Rv + Tv}"  # check energy conservation
    assert np.allclose(Rh + Th, 1), f"Energy conservation error in Rh + Th: {Rh + Th}"  # check energy conservation

    return rv, rh, tv, th, Rv, Rh, Tv, Th, mu2


# use the best function for the fresnel coefficients
fresnel_reflection_coefficients = fresnel_reflection_coefficients_maezawa09_rigorous_compiled


def snell_angle(eps_1, eps_2, mu1):
    """Compute mu2 the cos(angle) in the second medium according to Snell's law.

    This is exact for lossless media and an approximation for lossy media in O(L^2) where L is the loss tangent of the
    medium. For lossy media, use :py:func:`exact_snell_angle`.

    Args:
        eps_1: permittivity of medium 1.
        eps_2: permittivity of medium 2.
        mu1: cosine zenith angle in medium 1.

    Returns:
        mu2: float
            cosine zenith angle in medium 2.
    """
    # incident wavenumber
    n1 = np.sqrt(eps_1)
    kiz2 = n1.real**2 * (1 - mu1**2)  # this the square of kiz = n1 * sin(theta)

    # Snell's law
    mu2 = np.sqrt(np.maximum(1 - kiz2 / (np.sqrt(eps_2).real) ** 2, 0.0))

    return mu2


def exact_snell_angle(eps_1, eps_2, mu1):
    """Cosine of the refraction angle in medium 2 (non-magnetic, lossy-to-lossy).

    Refs: Guimarães, L. G., & Sampaio, E. E. S. (2008). A note on Snell laws for electromagnetic plane waves in lossy
    media. Journal of Quantitative Spectroscopy and Radiative Transfer, 109(11),
    2124–2140. https://doi.org/10.1016/j.jqsrt.2008.03.014

    Angles are the real energy-flow (Poynting) directions; the tangential wavenumber kiz is conserved and real. F
    or TE (h) the Poynting direction equals the phase direction; for TM (v) it differs slightly.

    Args:
        eps_1: complex permittivity of medium 1 (Re > 0, Im >= 0).
        eps_2: complex permittivity of medium 2 (Re > 0, Im >= 0).
        mu1:   cosine of the zenith angle in medium 1.

    Returns:
        mu2_h, mu2_v: cosine of the zenith angle in medium 2 (0 if the wave is
             evanescent, NaN for pol="v" if no real angle exists in medium 1,
             i.e. mu1 <= L1 / sqrt(1 + L1**2)).
    """
    eps_1 = np.asarray(eps_1, dtype=complex)
    eps_2 = np.asarray(eps_2, dtype=complex)

    mu1_squared = mu1**2
    sin2 = 1.0 - mu1_squared  # sin^2(theta1)
    L1_squared = (eps_1.imag / eps_1.real) ** 2  # loss tangent of medium 1 (squared)

    # kiz^2 = Re(eps_1) * kappa^2 * sin^2(theta1)

    kappa1_h_squared = 0.5 * (1.0 + np.sqrt(1.0 + L1_squared / mu1_squared))

    tan2 = sin2 / mu1_squared
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa1_v_squared = 0.5 * (1.0 + L1_squared) * (1.0 + np.sqrt((1.0 + L1_squared) / (1.0 - L1_squared * tan2)))

    kiz_h_squared = eps_1.real * kappa1_h_squared * sin2
    kiz_v_squared = eps_1.real * kappa1_v_squared * sin2

    L2_squared = (eps_2.imag / eps_2.real) ** 2  # loss tangent of medium 2 (squared)

    # 1st order H
    mu2_h_squared = 1.0 - kiz_h_squared / eps_2.real
    # 2nd order H
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa2_h_squared = 0.5 * (1.0 + np.sqrt(1.0 + L2_squared / mu2_h_squared))
        mu2_h_squared = np.maximum(1.0 - kiz_h_squared / (eps_2.real * kappa2_h_squared), 0.0)

    # 1st order V
    mu2_v_squared = 1.0 - kiz_v_squared / eps_2.real
    # 2nd order V
    with np.errstate(invalid="ignore", divide="ignore"):
        tan2 = 1 / mu2_v_squared - 1
        kappa2_v_squared = 0.5 * (1.0 + L2_squared) * (1.0 + np.sqrt((1.0 + L2_squared) / (1.0 - L2_squared * tan2)))
        mu2_v_squared = np.maximum(1.0 - kiz_v_squared / (eps_2.real * kappa2_v_squared), 0.0)

    return np.sqrt(mu2_h_squared), np.sqrt(mu2_v_squared)


def brewster_angle(eps_1, eps_2):
    """Compute the brewster angle.

    Args:
        eps_1: permittivity of medium 1.
        eps_2: permittivity of medium 2.

    Returns:
        angle: float
            brewster angle in radians.
    """
    return np.arctan(np.sqrt(eps_2.real / eps_1.real))


def fresnel_matrix(eps_1, eps_2, mu1, npol) -> tuple[smrt_matrix, smrt_matrix, np.ndarray]:
    """Compute the fresnel power reflection and transmission matrix for/in medium 1 laying above medium 2.

    Args:
        eps_1: permittivity of medium 1.
        eps_2: permittivity of medium 2.
        mu1: cosine zenith angle in medium 1.
        npol: number of polarizations to return.

    Returns:
        reflection_coefficients: matrix
            reflection coefficients matrix.
        transmission_coefficients: matrix
            transmission coefficients matrix.
    """
    mu1 = np.atleast_1d(mu1)
    assert len(mu1.shape) == 1  # 1D array

    rv, rh, mu2 = fresnel_reflection_coefficients_maezawa09_rigorous_compiled(eps_1, eps_2, mu1)

    reflection_coefficients = smrt_matrix.ones((npol, len(mu1)))
    transmission_coefficients = smrt_matrix.zeros((npol, len(mu1)))

    reflection_coefficients[0] = abs2(rv)
    reflection_coefficients[1] = abs2(rh)

    transmission_coefficients[0] = 1 - abs2(rv)
    transmission_coefficients[1] = 1 - abs2(rh)

    if npol >= 3:
        reflection_coefficients[2] = (rv * np.conj(rh)).real  # TsangI  Eq 7.2.93
        transmission_coefficients[2] = mu2 / mu1 * ((1 + rv) * np.conj(1 + rh)).real  # TsangI  Eq 7.2.95
        # It is not sure this equation is valid for strongly lossy materials

    return reflection_coefficients, transmission_coefficients, mu2


def fresnel_reflection_matrix(eps_1, eps_2, mu1, npol) -> tuple[smrt_matrix, np.ndarray]:
    """Compute the fresnel power reflection matrix for/in medium 1 laying above medium 2.

    Args:
        eps_1: permittivity of medium 1.
        eps_2: permittivity of medium 2.
        mu1: cosine zenith angle in medium 1.
        npol: number of polarizations to return.

    Returns:
        reflection_coefficients: matrix
            power reflection coefficients matrix.
    """
    mu1 = np.atleast_1d(mu1)
    assert len(mu1.shape) == 1  # 1D array

    reflection_coefficients = smrt_matrix.ones((npol, len(mu1)))

    rv, rh, mu2 = fresnel_reflection_coefficients_maezawa09_rigorous_compiled(eps_1, eps_2, mu1)

    reflection_coefficients[0] = abs2(rv)
    reflection_coefficients[1] = abs2(rh)

    if npol >= 3:
        reflection_coefficients[2] = (rv * np.conj(rh)).real  # TsangI  Eq 7.2.93
        # It is not sure this equation is valid for strongly lossy materials

    return reflection_coefficients, mu2


def fresnel_transmission_matrix(eps_1, eps_2, mu1, npol) -> tuple[smrt_matrix, np.ndarray]:
    """Compute the fresnel power transmission matrix for/in medium 1 lying above medium 2.

    Args:
        eps_1: permittivity of medium 1.
        eps_2: permittivity of medium 2.
        mu1: cosine zenith angle in medium 1.
        npol: number of polarizations to return.

    Returns:
        transmission_coefficients: matrix
            power transmission coefficients matrix.
    """
    mu1 = np.atleast_1d(mu1)
    assert len(mu1.shape) == 1  # 1D array

    transmission_coefficients = smrt_matrix.zeros((npol, len(mu1)))

    rv, rh, mu2 = fresnel_reflection_coefficients_maezawa09_rigorous_compiled(eps_1, eps_2, mu1)

    transmission_coefficients[0] = 1 - abs2(rv)
    transmission_coefficients[1] = 1 - abs2(rh)
    if npol >= 3:
        transmission_coefficients[2] = mu2 / mu1 * ((1 + rv) * np.conj(1 + rh)).real  # TsangI  Eq 7.2.95
        # It is not sure this equation is valid for strongly lossy materials

    if npol == 4:
        raise Exception("to be implemented, the matrix is not diagonal anymore")

    return transmission_coefficients, mu2


def field_fresnel_matrix(eps_1, eps_2, mu1) -> tuple[smrt_matrix, smrt_matrix, np.ndarray]:
    """Compute the fresnel field reflection and transmission matrices for/in medium 1 laying above medium 2.

    Args:
        eps_1: permittivity of medium 1.
        eps_2: permittivity of medium 2.
        mu1: cosine zenith angle in medium 1.

    Returns:
        reflection_matrix: matrix
            field reflection coefficients matrix.
        transmission_matrix: matrix
            field transmission coefficients matrix.
    """
    mu1 = np.atleast_1d(mu1)
    assert len(mu1.shape) == 1  # 1D array

    rv, rh, tv, th, mu2 = fresnel_coefficients_maezawa09_rigorous_compiled(eps_1, eps_2, mu1)

    # # check energy conservation # eq 55 in M09
    # assert np.allclose(th - rh, 1), f"Energy conservation violated in transmission {th=} and {rh=}"
    # # check energy conservation # eq 56 in M09
    # n1 = np.sqrt(eps_1)
    # n2 = np.sqrt(eps_2)
    # assert np.allclose(n2 * tv - n1.conj() * rv, n1), f"Energy conservation violated in transmission {tv=} and {rv=}"
    # see test_fresnel.py for more details

    return smrt_matrix(np.array((rv, rh))), smrt_matrix(np.array((tv, th))), mu2
