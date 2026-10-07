import numpy as np

from smrt.core.fresnel import fresnel_coefficients_maezawa09_rigorous_compiled


def test_fresnel_energy_conservation():
    """Test that the Fresnel coefficients satisfy energy conservation."""

    eps_1 = 2.25 + 0.1j  # Example permittivity for medium 1
    eps_2 = 3.0 + 0.7j  # Example perm

    mu1 = np.array([0.5, 0.7, 0.9])  # Example array of cosine of incident angles

    rv, rh, tv, th, _ = fresnel_coefficients_maezawa09_rigorous_compiled(eps_1, eps_2, mu1)

    # check energy conservation # eq 55 in M09
    assert np.allclose(th - rh, 1), f"Energy conservation violated in transmission {th=} and {rh=}"
    # check energy conservation # eq 56 in M09
    n1 = np.sqrt(eps_1)
    n2 = np.sqrt(eps_2)
    assert np.allclose(n2 * tv - n1.conj() * rv, n1), f"Energy conservation violated in transmission {tv=} and {rv=}"
