

import numpy as np
import scipy.integrate

from nose.tools import ok_


def phase_fn_int(phi, inc_pol, em, mu, npol):
    # inc gives the start row index within the phase matrix P to look at a pair of rows (single incidence angle geometry)
    # phi represents the difference in azimuth, with phi' = zero for incidence angle
    # integral will be performed over phi

    def phase_fn(mu):

        return em.ft_even_phase(0, mu, mu, npol=npol)

    p11_12 = phase_fn(mu)[0::2, inc_pol] + phase_fn(mu)[1::2, inc_pol]

    return scipy.integrate.simps(p11_12, mu)


# Generic test: check integral of phase function equals scattering coefficient
def test_energy_conservation(em, tolerance_pc, npol=None, subset=16):
    """test energy conservation

    :param em: the electromagnetic model that has been set up
    :param tolerance_pc: relative tolerance
"""

    # Default test is for 2 pol matrix
    if npol is None:
        npol = 2

    mu = np.linspace(-1, 1, 128)

    for inc_pol in range(len(mu))[::subset]:

        print ('Test failed for incidence__polarization_angle_number: ', inc_pol)  # Only prints if test fails

        # Put phase information back - arbitrary phi difference is zero so cos dphi, cos (2*dphi) = 1
        p_sum, p_err = scipy.integrate.quad(phase_fn_int, 0, 2. * np.pi, args=(inc_pol, em, mu, npol))
        phase_integral = p_sum / (4. * np.pi)
        print ('Integrated phase function is: ', phase_integral)  # Only prints if test fails
        print ('Scattering coefficient is: ', em.ks)  # Only prints if test fails
        ok_(abs(phase_integral - em.ks) < tolerance_pc * em.ks)
