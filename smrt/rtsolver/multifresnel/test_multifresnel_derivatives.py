import pytest
import numpy as np
from smrt.permittivity import ice
from smrt.core.globalconstants import C_SPEED
from smrt.rtsolver.multifresnel import multifresnel
from smrt.rtsolver.multifresnel.multifresnel_derivatives import (
    complex_polarized_id23,
    forward_matrix_derivative,
    get_optical_depth_derivative,
    dAdTk,
    dinvAdTk,
    forward_matrix_coefficient_with_r_derivative,
)


def test_complex_polarized_id23():
    I2 = complex_polarized_id23(2)
    expected_I2 = np.array(
        [[[[1, 1], [1, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[0, 0], [0, 0]]]]
    )
    np.testing.assert_allclose(I2, expected_I2)


@pytest.mark.parametrize(
    ["temperature", "frequency", "theta"], [(200, 1.4e9, 30), (220, 2e9, 30), (240, 0.4e9, 40), (260, 1.4e9, 40)]
)
def test_optical_depth_derivative(temperature, frequency, theta):
    dt = 1e-7
    eps1: complex = 1
    eps2: complex = ice.ice_permittivity_maetzler06(temperature=temperature, frequency=frequency)
    eps3: complex = ice.ice_permittivity_maetzler06(temperature=temperature + dt, frequency=frequency)
    mu: float = np.atleast_1d(np.cos(theta))
    _, _, mu2 = multifresnel.fresnel_coefficients_maezawa09_rigorous(eps1, eps2, mu)
    _, _, mu3 = multifresnel.fresnel_coefficients_maezawa09_rigorous(eps1, eps3, mu)
    kd: float = 2 * np.pi * frequency / C_SPEED * 10
    dtau = get_optical_depth_derivative(
        kd=kd, frac_volume=1, temperature=temperature, frequency=frequency, eps2=eps2, mu2=mu2
    )
    taup = 2 * np.sqrt(eps3).imag * kd / mu3
    tau = 2 * np.sqrt(eps2).imag * kd / mu2
    dtauincr = (taup - tau) / dt
    np.testing.assert_allclose(dtau, dtauincr, rtol=1e-2)


@pytest.mark.parametrize(
    ["temperature", "frequency", "theta"], [(200, 1.4e9, 30), (220, 2e9, 30), (240, 0.4e9, 40), (260, 1.4e9, 40)]
)
def test_dAdtk(temperature, frequency, theta):
    dt = 1e-7
    eps1: complex = 1
    eps2: complex = ice.ice_permittivity_maetzler06(temperature=temperature, frequency=frequency)
    eps3: complex = ice.ice_permittivity_maetzler06(temperature=temperature + dt, frequency=frequency)
    mu: float = np.atleast_1d(np.cos(theta))
    _, _, mu2 = multifresnel.fresnel_coefficients_maezawa09_rigorous(eps1, eps2, mu)
    _, _, mu3 = multifresnel.fresnel_coefficients_maezawa09_rigorous(eps1, eps3, mu)
    kd: float = 2 * np.pi * frequency / C_SPEED * 10
    dA = dAdTk(kd=kd, frac_volume=1, temperature=temperature, frequency=frequency, eps2=eps2, mu2=mu2)
    optical_depth_p = 2 * np.sqrt(eps3).imag * kd / mu3
    Ap = np.exp(-optical_depth_p)[None, :]
    optical_depth = 2 * np.sqrt(eps2).imag * kd / mu2
    A = np.exp(-optical_depth)[None, :]
    dAincr = (Ap - A) / dt
    np.testing.assert_allclose(dA, dAincr, rtol=1e-2)


@pytest.mark.parametrize(
    ["temperature", "frequency", "theta"], [(200, 1.4e9, 30), (220, 2e9, 30), (240, 0.4e9, 40), (260, 1.4e9, 40)]
)
def test_dinvAdtk(temperature, frequency, theta):
    dt = 1e-7
    eps1: complex = 1
    eps2: complex = ice.ice_permittivity_maetzler06(temperature=temperature, frequency=frequency)
    eps3: complex = ice.ice_permittivity_maetzler06(temperature=temperature + dt, frequency=frequency)
    mu: float = np.atleast_1d(np.cos(theta))
    _, _, mu2 = multifresnel.fresnel_coefficients_maezawa09_rigorous(eps1, eps2, mu)
    _, _, mu3 = multifresnel.fresnel_coefficients_maezawa09_rigorous(eps1, eps3, mu)
    kd: float = 2 * np.pi * frequency / C_SPEED * 10
    dinvA = dinvAdTk(kd=kd, frac_volume=1, temperature=temperature, frequency=frequency, eps2=eps2, mu2=mu2)
    optical_depth_p = 2 * np.sqrt(eps3).imag * kd / mu3
    invAp = np.exp(optical_depth_p)[None, :]
    optical_depth = 2 * np.sqrt(eps2).imag * kd / mu2
    invA = np.exp(optical_depth)[None, :]
    dinvAincr = invAp / dt - invA / dt
    print(f"dinvaincr = {dinvAincr}")
    np.testing.assert_allclose(dinvA, dinvAincr, rtol=1e-2)


# @pytest.mark.skip(reason="Test fails for now")
@pytest.mark.parametrize(
    ["temperature", "frequency", "theta"], [(200, 1.4e9, 30), (220, 2e9, 30), (240, 0.4e9, 40), (260, 1.4e9, 40)]
)
def test_forward_matrix_derivative_coefficient(temperature, frequency, theta):
    dt = 1e-7
    eps1: complex = 1
    eps2: complex = ice.ice_permittivity_maetzler06(temperature=temperature, frequency=frequency)
    eps3: complex = ice.ice_permittivity_maetzler06(temperature=temperature + dt, frequency=frequency)
    mu: float = np.atleast_1d(np.cos(theta))
    kd: float = 2 * np.pi * frequency / C_SPEED * 10

    dM = forward_matrix_derivative(
        eps1=eps1, eps2=eps2, mu=mu, kd=kd, temperature=temperature, frequency=frequency, frac_volume=1
    )[0, 0]
    Mp, _ = multifresnel.forward_matrix_fulloutput(eps1=eps1, eps2=eps3, mu=mu, kd=kd, temperature=(temperature + dt))
    M, _ = multifresnel.forward_matrix_fulloutput(eps1, eps2, mu, kd, temperature=temperature)
    dMincr = Mp[0, 0] / dt - M[0, 0] / dt
    np.testing.assert_allclose(dM, dMincr, rtol=1e-2)


# @pytest.mark.skip(reason="The independant r hypothesis seems correct")
@pytest.mark.parametrize(
    ["temperature", "frequency", "theta"], [(200, 1.4e9, 30), (220, 2e9, 30), (240, 0.4e9, 40), (260, 1.4e9, 40)]
)
def test_forward_matrix_coefficient_with_r_derivative(temperature, frequency, theta):
    dt = 1e-6
    eps1: complex = 1
    eps2: complex = ice.ice_permittivity_maetzler06(temperature=temperature, frequency=frequency)
    eps3: complex = ice.ice_permittivity_maetzler06(temperature=temperature + dt, frequency=frequency)
    mu: float = np.atleast_1d(np.cos(theta))
    kd: float = 2 * np.pi * frequency / C_SPEED * 10

    dMoo = forward_matrix_coefficient_with_r_derivative(
        eps1=eps1, eps2=eps2, mu=mu, kd=kd, temperature=temperature, frequency=frequency, frac_volume=1
    )
    Mp, _ = multifresnel.forward_matrix_fulloutput(eps1=eps1, eps2=eps3, mu=mu, kd=kd, temperature=(temperature + dt))
    M, _ = multifresnel.forward_matrix_fulloutput(eps1, eps2, mu, kd, temperature=temperature)
    # print(f"M = {M}\nMp-M = {Mp - M}")
    dMincr = (Mp - M) / dt
    np.testing.assert_allclose(dMoo, dMincr[0, 0], rtol=1e-2)
