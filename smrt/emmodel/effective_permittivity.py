# coding: utf-8

"""
This module contains functions that are not specific to a particular electromagnetic model
and are available to be imported by any electromagnetic model. It is the responsibility of the
developer to ensure these functions, if used, are appropriate and consistent with the physics of the electromagnetic model.
"""

# Import statements
import itertools
import numpy as np
from smrt.core.error import  SMRTError

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


def polder_van_santen(frac_volume, e0=None, eps=None, depol_xyz=None, inclusion_shape=None, mixing_ratio=1):
    """ Calculates effective permittivity of snow by solution of quadratic Polder Van Santen equation for spherical inclusion.

    :param frac_volume: Fractional volume of inclusions
    :param e0: Permittivity of background (default is 1)
    :param eps: Permittivity of scattering material (default is 3.185 to compare with MEMLS)
    :param depol_xyz: [Optional] Depolarization factors, spherical isotropy is default. It is not taken into account here.
    :param inclusion_shape: Assumption for shape(s) of brine inclusions. Can be a string for single shape, or a list/tuple/dict of strings for mixture of shapes. So far, we have the following shapes: "spheres" and "random_needles" (i.e. randomly-oriented elongated ellipsoidal inclusions). 
            If the argument is a dict, the keys are the shapes and the values are the mixing ratio. If it is a list, the mixing_ratio argument is required.
    :param mixing_ratio: The mixing ratio of the shapes. This is only relevant when inclusion_shape is a list/tuple. Mixing ratio must be a sequence with length len(inclusion_shape)-1. The mixing ratio of the last shapes is deduced as the sum of the ratios must equal to 1.
    :returns: Effective permittivity

    **Usage example:**

    ::

        from .commonfunc import polder_van_santen
        effective_permittivity = polder_van_santen(frac_volume, e0, eps)

        # for a mixture of 30% spheres and 70% needles
        effective_permittivity = polder_van_santen(frac_volume, e0, eps, inclusion_shape={"spheres": 0.3, "random_needles": 0.7})
        # or
        effective_permittivity = polder_van_santen(frac_volume, e0, eps, inclusion_shape=("spheres", "random_needles"), mixing_ratio=0.3)

    .. todo::

        Extend Polder Van Santen model to account for ellipsoidal inclusions

    """

    if inclusion_shape is not None and not isinstance(inclusion_shape, str):
        # then it is a sequence or dict
        if  isinstance(inclusion_shape, dict):

            mixing_ratio = inclusion_shape.values()
            inclusion_shape = inclusion_shape.keys()

        else:
            # we've a sequence let's iterate over it

            try:
                len(mixing_ratio)
            except:
                # here -> we must have a number, let's make a tuple with length 1
                mixing_ratio = (float(mixing_ratio),)

            if len(mixing_ratio) == len(inclusion_shape) - 1:
                mixing_ratio = list(mixing_ratio) + [1 - np.sum(mixing_ratio)]
            elif len(mixing_ratio) != len(inclusion_shape):
                raise SMRTError("The length of inclusion_shape and mixing_ratio are incompatible. See the documentation.")

        return sum((mixing * polder_van_santen(frac_volume, e0=e0, eps=eps, depol_xyz=depol_xyz, inclusion_shape=shape) for shape, mixing in zip(inclusion_shape, mixing_ratio)))

    if e0 is None:
        e0 = 1.

    if eps is None:
        eps = 3.185  # MEMLS default for PVS calculation

    # Polder Van Santen / de Loor / Böttcher / Bruggeman formula
    # Solution of quadratic equation arising from eqn 9.2. in Sihvola: Electromagnetic Mixing Formulas and Applications
    if (inclusion_shape is None) or (inclusion_shape == "spheres"):
        a_quad = 2.
        b_quad = eps - 2 * e0 - 3. * frac_volume * (eps - e0)
        c_quad = - eps * e0

    # Polder and Van Santen model, modified by de Loor (according to Shokr (1998) simplified by Hoekstra and Capillino (1971))
    # Solution of quadratic equation arising from eqn (18) in Shokr (1998): 'Field Observations and Model Calculations of Dielectric Properties of Arctic Sea Ice in the Microwave C-Band', IEEE.
    elif inclusion_shape == "random_needles":
        a_quad = 1.
        b_quad = eps - e0 - 5. / 3. * frac_volume * (eps - e0)
        c_quad = - eps * (e0 + 1. / 3. * frac_volume * (eps - e0))

    else:
        raise SMRTError("inclusion_shape must be one of (or a list of) the following: 'spheres' (default) or 'random_needles'.")
   
    return (-b_quad + np.sqrt(b_quad**2 - 4. * a_quad * c_quad)) / (2. * a_quad)


# synonym
bruggeman = polder_van_santen


def maxwell_garnett(frac_volume, e0, eps, depol_xyz=None, inclusion_shape=None):
    """ Calculates effective permittivity of snow by solution of Maxwell-Garnett equation.

    :param frac_volume: Fractional volume of snow
    :param e0: Permittivity of background (no default, must be provided)
    :param eps: Permittivity of scattering material (no default, must be provided)
    :param depol_xyz: [Optional] Depolarization factors, spherical isotropy is default
    :added **kwargs to enable same structure of calling maxwell_garnett() and polder_van_santen() as effective_permittivity_model...
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

    if inclusion_shape is not None and inclusion_shape != "spheres":
        raise SMRTError("inclusion_shape must be set to 'spheres'")

    if depol_xyz is None:
        depol_xyz = np.array([1. / 3.] * 3)

    # Calculate x, y, z components of effective permittivity from Maxwell-Garnett theory
    effective_permittivity_xyz = e0 + e0 * frac_volume * (eps - e0) / (e0 + (1. - frac_volume) * depol_xyz * (eps - e0))

    # Assume random orientation i.e. 1/3 of each polarizability component provides equal shares
    # to the macroscopic polarization density
    # See pg 68 Sihvola: Electromagnetic mixing formulas and applications

    return np.mean(effective_permittivity_xyz, dtype=np.complex128)
