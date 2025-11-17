import numpy as np
import scipy.integrate


# Generic test: check integral of phase function equals scattering coefficient
def test_energy_conservation(em, tolerance_pc, npol=None, subset=16):
    """
    test energy conservation

    Args:
        em: the electromagnetic model that has been set up
        tolerance_pc: relative tolerance
        npol: (Default value = None)
        subset: (Default value = 16)
    """

    # Default test is for 2 pol matrix
    if npol is None:
        npol = 2

    mu = np.linspace(-1, 1, 128)

    ft_even_phase = em.ft_even_phase(mu, mu, m_max=0, npol=npol)

    for pol in [0, 1]:
        for inc in range(len(mu))[::subset]:
            # Put phase information back - arbitrary phi difference is zero so cos dphi, cos (2*dphi) = 1
            # p_sum, p_err = scipy.integrate.quad(phase_fn_int, 0, 2. * np.pi, args=(inc, pol, em, mu, npol))
            # not needed as we already use m=0

            p11_12 = np.sum(ft_even_phase[:, pol, 0, :, inc], axis=0)
            p_sum = 2 * np.pi * scipy.integrate.simpson(p11_12, x=mu)
            phase_integral = p_sum / (4.0 * np.pi)

            ks = em.ks(mu[inc]).values

            print("Test failed for incidence__polarization_angle_number: ", inc, pol)  # Only prints if test fails
            print("Integrated phase function is: ", phase_integral)  # Only prints if test fails
            print("Scattering coefficient is: ", ks)  # Only prints if test fails
            assert np.allclose(ks, phase_integral, rtol=tolerance_pc)
