# coding: utf-8

"""
This module contains functions that are not specific to a particular electromagnetic model
and are available to be imported by any electromagnetic model. It is the responsibility of the
developer to ensure these functions, if used, are appropriate and consistent with the physics of the electromagnetic model.
"""

# Import statements
import numpy as np

def depolarization_factors(length_ratio=None):
    """ Calculates depolarization factors for use in effective permittivity models. These
    are a measure of the anisotropy of the snow. Default is spherical isotropy.

    :param length_ratio: [Optional] ratio of microstructure length measurement in x/y direction to z-direction [unitless].
    :returns: [x, y, z] depolarization factor array

    **Usage example:**

    ::

        # If imported to an electromagnetic model:
        from .commonfunc import depolarization_factors
        depol_xyz = depolarization_factors()

        # General import:
        from smrt.emmodel.commonfunc import depolarization_factors
        depol_xyz = depolarization_factors(length_ratio=1.2)

    """

    # If a length ratio is not specified, assumes spherical isotropy
    if length_ratio is None:
        length_ratio = 1.

    # Calculation of anisotropy factor
    if length_ratio == 1:
        anisotropy_q = 1. / 3.
    elif length_ratio > 1:
        # Upper Equation 4 from Löwe et al. TC (2013)
        chi_b = np.sqrt(1. - 1. / (length_ratio**2.))
        ln_term = np.log((1. + chi_b) / (1. - chi_b))
        anisotropy_q = 0.5 * (1. + (1. / (length_ratio**2. - 1.)) * (1. - (1. / (2. * chi_b)) * ln_term))
    else:
        # Lower Equation 4 from Löwe et al. TC (2013)
        chi_a = np.sqrt(1. / length_ratio**2. - 1.)
        anisotropy_q = 0.5 * (1. + (1. / (length_ratio**2. - 1.)) * (1. - (1. / chi_a) * np.arctan(chi_a)))

    return np.array([anisotropy_q, anisotropy_q, (1. - 2. * anisotropy_q)])


def polder_van_santen(frac_volume, e0=None, eps=None, depol_xyz=None):
    """ Calculates effective permittivity of snow by solution of quadratic Polder Van Santen equation for spherical inclusion.

    :param frac_volume: Fractional volume of snow
    :param e0: Permittivity of background (default is 1)
    :param eps: Permittivity of scattering material (default is 3.185 to compare with MEMLS)
    :param depol_xyz: [Optional] Depolarization factors, spherical isotropy is default. It is not taken into account here.
    :returns: Effective permittivity

    **Usage example:**

    ::

        from .commonfunc import polder_van_santen
        effective_permittivity = polder_van_santen(frac_volume, e0, eps)

    .. todo::

        Extend Polder Van Santen model to account for ellipsoidal inclusions

    """

    if e0 is None:
        e0 = 1.

    if eps is None:
        eps = 3.185  # MEMLS default for PVS calculation

    # Polder Van Santen / de Loor / Böttcher / Bruggeman formula
    # Solution of quadratic equation arising from eqn 9.2. in Sihvola: Electromagnetic Mixing Formulas and Applications
    a_quad = 2.
    b_quad = eps - 2 * e0 - 3. * frac_volume * (eps - e0)
    c_quad = - eps * e0

    return (-b_quad + np.sqrt(b_quad**2 - 4. * a_quad * c_quad)) / (2. * a_quad)


# synonym
bruggeman = polder_van_santen


def maxwell_garnett(frac_volume, e0, eps, depol_xyz=None):
    """ Calculates effective permittivity of snow by solution of Maxwell-Garnett equation.

    :param frac_volume: Fractional volume of snow
    :param e0: Permittivity of background (no default, must be provided)
    :param eps: Permittivity of scattering material (no default, must be provided)
    :param depol_xyz: [Optional] Depolarization factors, spherical isotropy is default
    :returns: random orientation effective permittivity

    **Usage example:**

    ::

        # If used by electromagnetic model module:
        from .commonfunc import maxwell_garnett
        effective_permittivity = maxwell_garnett(frac_volume=0.2, 
                                                 e0=1, 
                                                 eps=3.185, 
                                                 depol_xyz=[0.3, 0.3, 0.4])

        # If accessed from elsewhere, use absolute import
        from smrt.emmodel.commonfunc import maxwell_garnett

    """

    if depol_xyz is None:
        depol_xyz = np.array([1. / 3.] * 3)

    # Calculate x, y, z components of effective permittivity from Maxwell-Garnett theory
    effective_permittivity_xyz = e0 + e0 * frac_volume * (eps - e0) / (e0 + (1. - frac_volume) * depol_xyz * (eps - e0))

    # Assume random orientation i.e. 1/3 of each polarizability component provides equal shares
    # to the macroscopic polarization density
    # See pg 68 Sihvola: Electromagnetic mixing formulas and applications

    return np.mean(effective_permittivity_xyz, dtype=np.complex128)
