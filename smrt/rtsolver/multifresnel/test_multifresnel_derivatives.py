import pytest
import numpy as np
from smrt.permittivity import ice
from smrt.core.globalconstants import C_SPEED
from smrt.rtsolver.multifresnel import multifresnel
from smrt.rtsolver.multifresnel.multifresnel_derivatives import complex_polarized_id23, forward_matrix_derivative


def test_complex_polarized_id23():
    I2 = complex_polarized_id23(2)
    expected_I2 = np.array(
        [[[[1, 1], [1, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[0, 0], [0, 0]]]]
    )
    np.testing.assert_allclose(I2, expected_I2)


@pytest.mark.parametrize(
    ["temperature", "frequency", "theta"], [(200, 1.4e9, 30), (220, 1.4e9, 30), (240, 0.4e9, 40), (260, 17e9, 40)]
)
def test_forward_matrix_derivative(temperature, frequency, theta):
    eps1: complex = 1
    eps2: complex = ice.ice_permittivity_maetzler06(temperature=temperature, frequency=frequency)
    mu: float = np.atleast_1d(np.cos(theta))
    kd: float = 2 * np.pi * frequency / C_SPEED * 10
    dt = 1e-6
    dM = forward_matrix_derivative(eps1=eps1, eps2=eps2, mu=mu, kd=kd, temperature=temperature, frequency=frequency)
    Mp, _ = multifresnel.forward_matrix_fulloutput(eps1=eps1, eps2=eps2, mu=mu, kd=kd, temperature=(temperature + dt))
    M, _ = multifresnel.forward_matrix_fulloutput(eps1, eps2, mu, kd, temperature=temperature)
    print(f"M = {M}\nMp-M = {Mp - M}")
    dMincr = (Mp - M) / dt
    np.testing.assert_allclose(dM, dMincr)
