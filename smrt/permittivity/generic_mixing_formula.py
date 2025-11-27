# coding: utf-8

"""
Provide functions that are not tied to a particular electromagnetic model
and are available to be imported by any electromagnetic model.

Note:
    It is the responsibility of the developer to ensure these functions, if used, are appropriate and consistent
    with the physics of the electromagnetic model.
"""

from collections.abc import Mapping, Sequence
from typing import Callable, Optional

# Import statements
import numpy as np
import scipy.optimize

from smrt.core.error import SMRTError
from smrt.core.layer import layer_properties

from .depolarization_factors import depolarization_factors_spheroids


@layer_properties(
    "frac_volume",
    optional_arguments=(
        "inclusion_shape",
        "depolarization_factors",
        "length_ratio",
        "mixing_ratio",
    ),
)
def polder_van_santen(
    frac_volume: float,
    e0: complex = 1,
    eps: complex = 3.185,
    depolarization_factors=None,
    length_ratio: Optional[float] = None,
    inclusion_shape: Optional[str | Sequence | Mapping] = None,
    mixing_ratio: Optional[int] = None,
):
    """
    Calculate the effective permittivity of snow by solution of quadratic Polder Van Santen equation for spherical
    or random needle inclusions only, potentially a mixing of these shapes (Sihvola, 1999).

    Note:
        See :py:func:`~smrt.permittivity.generic_mixing_formula.general_polder_van_santen` for an alternative
        function (more generic, but slower).

    Args:
        frac_volume: Fractional volume of inclusions.
        e0: Permittivity of background (default is 1).
        eps: Permittivity of scattering material (default is 3.185 to compare with MEMLS)
        depolarization_factors: [Optional] Depolarization factors, spherical isotropy is default. It is not taken into account here.
        length_ratio: Length_ratio. Used to estimate depolarization factors when they are not given.
        inclusion_shape: Assumption for shape(s) of brine inclusions. Can be a string for single shape, or a list/tuple/dict of strings
            for mixture of shapes. So far, we have the following shapes: "spheres" and "random_needles" (i.e. randomly-oriented elongated
            ellipsoidal inclusions). If the argument is a dict, the keys are the shapes and the values are the mixing ratio. If it is
            a list, the mixing_ratio argument is required.
        mixing_ratio: The mixing ratio of the shapes. This is only relevant when inclusion_shape is a list/tuple. Mixing ratio must
            be a sequence with length len(inclusion_shape)-1. The mixing ratio of the last shapes is deduced as the sum of the ratios
            must equal to 1.

    Returns:
        Effective permittivity

    Usage:
        >>> from smrt.permittivity.generic_mixing_formula import polder_van_santen
        >>> effective_permittivity = polder_van_santen(frac_volume, e0, eps)

        for a mixture of 30% spheres and 70% needles
        >>> effective_permittivity = polder_van_santen(frac_volume, e0, eps, inclusion_shape={"spheres": 0.3, "random_needles": 0.7})

        or
        >>> effective_permittivity = polder_van_santen(frac_volume, e0, eps, inclusion_shape=("spheres", "random_needles"), mixing_ratio=0.3)

    todo:
        Extend Polder Van Santen model to account for ellipsoidal inclusions

    References:
        Sihvola, A.: Electromagnetic Mixing Formulas and Applications, 1999, INSTITUTION OF ENGINEERING & T
        http://www.ebook.de/de/product/21470462/a_sihvola_electromagnetic_mixing_formulas_and_applications.html

    """

    # first deal with the case inclusion_shape is a dict or a Sequence
    if inclusion_shape is not None and not isinstance(inclusion_shape, str):
        # then it is a sequence or dict
        if isinstance(inclusion_shape, dict):
            if mixing_ratio is not None:
                raise SMRTError("Setting mixing_ratio and using a dict for inclusion_shape is ambiguous.")
            mixing_ratio = inclusion_shape.values()
            inclusion_shape = inclusion_shape.keys()

        # we've a sequence let's iterate over it

        try:
            len(mixing_ratio)
        except TypeError:
            # here -> we must have a number, let's make a tuple with length 1
            mixing_ratio = (float(mixing_ratio),)

        if len(mixing_ratio) == len(inclusion_shape) - 1:
            mixing_ratio = list(mixing_ratio) + [1 - np.sum(mixing_ratio)]
        elif len(mixing_ratio) != len(inclusion_shape):
            raise SMRTError("The length of inclusion_shape and mixing_ratio are incompatible. See the documentation.")

        return sum(
            (
                mixing * polder_van_santen(frac_volume, e0=e0, eps=eps, inclusion_shape=shape)
                for shape, mixing in zip(inclusion_shape, mixing_ratio)
            )
        )

    assert np.all(frac_volume <= 1), f"the fractional volume is larger than 1: {frac_volume:g}"

    # Polder Van Santen / de Loor / Böttcher / Bruggeman formula
    # Solution of quadratic equation arising from eqn 9.2. in Sihvola: Electromagnetic Mixing Formulas and Applications
    if (inclusion_shape is None) or (inclusion_shape == "spheres"):
        a_quad = 2.0
        b_quad = eps - 2 * e0 - 3.0 * frac_volume * (eps - e0)
        c_quad = -eps * e0

    # Polder and Van Santen model, modified by de Loor (according to Shokr (1998) simplified by Hoekstra and Capillino (1971))
    # Solution of quadratic equation arising from eqn (18) in Shokr (1998): 'Field Observations and Model Calculations of Dielectric Properties of Arctic Sea Ice in the Microwave C-Band', IEEE.
    elif inclusion_shape == "random_needles":
        a_quad = 1.0
        b_quad = eps - e0 - 5.0 / 3.0 * frac_volume * (eps - e0)
        c_quad = -eps * (e0 + 1.0 / 3.0 * frac_volume * (eps - e0))

    else:
        raise SMRTError(
            "inclusion_shape must be one of (or a list of) the following: 'spheres' (default) or 'random_needles'."
        )

    return (-b_quad + np.sqrt(b_quad**2 - 4.0 * a_quad * c_quad)) / (2.0 * a_quad)


@layer_properties(
    "frac_volume",
    optional_arguments=("inclusion_shape", "depolarization_factors", "length_ratio"),
)
def general_polder_van_santen(
    frac_volume: float,
    e0: complex = 1,
    eps: complex = 3.185,
    depolarization_factors=None,
    length_ratio: Optional[float] = None,
    inclusion_shape: Optional[str] = None,
):
    """
    Calculate the effective permittivity of snow by solution of quadratic Polder Van Santen equation.

    Note:
        For spherical or random needle inclusions only, potentially a mixing of these shapes (Sihvola, 1999).
        See :py:func:`~smrt.permittivity.generic_mixing_formula.polder_van_santen` for an alternative function (faster but
        specific to some shapes).

    Args:
        frac_volume: Fractional volume of inclusions.
        e0: Permittivity of background (default is 1).
        eps: Permittivity of scattering material (default is 3.185 to compare with MEMLS).
        depolarization_factors: [Optional] Depolarization factors, spherical isotropy is default. It is not taken into account here.
        length_ratio: Length_ratio. Used to estimate depolarization factors when they are not given.
        inclusion_shape: Assumption for shape(s) of brine inclusions. Can be a string for single shape, or a list/tuple/dict of strings
            for mixture of shapes. So far, we have the following shapes: "spheres" and "random_needles" (i.e. randomly-oriented elongated
            ellipsoidal inclusions). If the argument is a dict, the keys are the shapes and the values are the mixing ratio. If it is a
            list, the mixing_ratio argument is required.
        mixing_ratio: The mixing ratio of the shapes. This is only relevant when inclusion_shape is a list/tuple. Mixing ratio must be
            a sequence with length len(inclusion_shape)-1. The mixing ratio of the last shapes is deduced as the sum of the ratios must
            equal to 1.

    Returns:
        Effective permittivity.

    References
        Sihvola, A.: Electromagnetic Mixing Formulas and Applications, 1999, INSTITUTION OF ENGINEERING & T
        http://www.ebook.de/de/product/21470462/a_sihvola_electromagnetic_mixing_formulas_and_applications.html
    """

    depol_xyz = _get_depolarization_factors(depolarization_factors, inclusion_shape, frac_volume, length_ratio)

    def pvs_equation(x):
        eps_eff0 = complex(x[0], x[1])

        eps_eff = e0 + frac_volume * (eps - e0) * eps_eff0 * np.mean(1 / (eps_eff0 + depol_xyz * (eps - eps_eff0)))

        residual = eps_eff0 - eps_eff
        return [residual.real, residual.imag]

    # first guess
    eps_eff0 = e0 * (1 - frac_volume) + eps * frac_volume
    res = scipy.optimize.root(pvs_equation, [eps_eff0.real, eps_eff0.imag])

    return complex(res.x[0], res.x[1])


# synonym
bruggeman = polder_van_santen


def polder_van_santen_three_spherical_components(f1, f2, eps0, eps1, eps2):
    """
    Calculate effective permittivity using Polder and van Santen with three components assuming spherical inclusions.

    Args:
        f1: fractional volume of component 1.
        f2: fractional volume of component 2.
        eps0: permittivity of material 0.
        eps1: permittivity of material 1.
        eps2: permittivity of material 2.

    Returns:
        Effective permittivity.

    References
        Sihvola, A.: Electromagnetic Mixing Formulas and Applications, 1999, INSTITUTION OF ENGINEERING & T
        http://www.ebook.de/de/product/21470462/a_sihvola_electromagnetic_mixing_formulas_and_applications.html
    """
    if (np.array(f1).ndim >= 1) or (np.array(f2).ndim >= 1):

        def func(f1, f2):
            return polder_van_santen_three_spherical_components(f1, f2, eps0, eps1, eps2)

        return np.vectorize(func)(f1, f2)

    # rough first guess
    f0 = 1 - f1 - f2
    # this first guess is not good enough: eps_eff0_0 = f1 * eps1 + f2 * eps2 + f0 * eps0
    eps_eff0 = polder_van_santen(f0, polder_van_santen(f2 / (f1 + f2), eps1, eps2), eps0)

    def pvs_equation(x):
        eps_eff = complex(x[0], x[1])
        residual = (
            eps_eff
            * (1 - 3 * f2 * (eps2 - eps0) / (2 * eps_eff + eps2) - 3 * f1 * (eps1 - eps0) / (2 * eps_eff + eps1))
            - eps0
        )
        return [residual.real, residual.imag]

    res = scipy.optimize.root(pvs_equation, [eps_eff0.real, eps_eff0.imag])

    eps_eff = complex(res.x[0], res.x[1])

    return eps_eff


def polder_van_santen_three_components(f1, f2, eps0, eps1, eps2, A1, A2):
    """
    Calculate effective permittivity using Polder and van Santen with three components

    Args:
        f1: fractional volume of component 1.
        f2: fractional volume of component 2.
        eps0: permittivity of material 0.
        eps1: permittivity of material 1.
        eps2: permittivity of material 2.
        A1: depolarization factor for material 1.
        A2: depolarization factor for material 2.

    Returns:
        Effective permittivity.

    References
        Sihvola, A.: Electromagnetic Mixing Formulas and Applications, 1999, INSTITUTION OF ENGINEERING & T
        http://www.ebook.de/de/product/21470462/a_sihvola_electromagnetic_mixing_formulas_and_applications.html
    """
    if (np.array(f1).ndim >= 1) or (np.array(f2).ndim >= 1):

        def func(f1, f2):
            return polder_van_santen_three_components(f1, f2, eps0, eps1, eps2, A1, A2)

        return np.vectorize(func)(f1, f2)

    # rough first guess
    f0 = 1 - f1 - f2
    eps_eff0 = f1 * eps1 + f2 * eps2 + f0 * eps0

    def pvs_equation(x):
        eps_eff = complex(x[0], x[1])
        residual = (
            eps_eff
            * (
                1
                - 1 / 3 * f2 * (eps2 - eps0) * sum(1 / (eps_eff + A2j * (eps2 - eps_eff)) for A2j in A2)
                - 1 / 3 * f1 * (eps1 - eps0) * sum(1 / (eps_eff + A1j * (eps1 - eps_eff)) for A1j in A1)
            )
            - eps0
        )
        return [residual.real, residual.imag]

    res = scipy.optimize.root(pvs_equation, [eps_eff0.real, eps_eff0.imag])

    eps_eff = complex(res.x[0], res.x[1])

    return eps_eff


@layer_properties(
    "frac_volume",
    optional_arguments=("inclusion_shape", "depolarization_factors", "length_ratio"),
)
def maxwell_garnett(
    frac_volume,
    e0,
    eps,
    depolarization_factors=None,
    inclusion_shape=None,
    length_ratio=None,
):
    """
    Calculate effective permittivity using Maxwell-Garnett equation.

    Args:
        frac_volume: Fractional volume of snow.
        e0: Permittivity of background (no default, must be provided).
        eps: Permittivity of scattering material (no default, must be provided).
        depolarization_factors: [Optional] Depolarization factors, spherical isotropy is default.
        length_ratio: Length_ratio. Used if depolarization factors are not given to estimate depolarization factors
            of spheroids when they are not given.
        inclusion_shape: Assumption for shape(s) of brine inclusions. Can be a string for single shape, or a
            list/tuple/dict of strings for mixture of shapes. So far, we have the following shapes: "spheres" and
            "random_needles" (i.e. randomly-oriented elongated ellipsoidal inclusions).
            If the argument is a dict, the keys are the shapes and the values are the mixing ratio. If it is a list, the
            mixing_ratio argument is required.

    Returns:
        random orientation effective permittivity.

    **Usage**::

        # If used by electromagnetic model module:
        from .commonfunc import maxwell_garnett
        effective_permittivity = maxwell_garnett(frac_volume=0.2, e0=1, eps=3.185, depol_xyz=[0.3, 0.3, 0.4])

        #If accessed from elsewhere, use absolute import
        from smrt.emmodel.commonfunc import maxwell_garnett
    """

    assert np.all(frac_volume <= 1)

    if inclusion_shape is not None and inclusion_shape != "spheres":
        raise SMRTError("inclusion_shape must be set to 'spheres'")

    depol_xyz = _get_depolarization_factors(depolarization_factors, inclusion_shape, frac_volume, length_ratio)

    # Calculate x, y, z components of effective permittivity from Maxwell-Garnett theory
    effective_permittivity_xyz = e0 * (
        1 + frac_volume * (eps - e0) / (e0 + (1.0 - frac_volume) * depol_xyz * (eps - e0))
    )

    # Assume random orientation i.e. 1/3 of each polarizability component provides equal shares
    # to the macroscopic polarization density
    # See pg 68 Sihvola: Electromagnetic mixing formulas and applications

    return np.mean(effective_permittivity_xyz, dtype=np.complex128)


@layer_properties("frac_volume")
def maxwell_garnett_for_spheres(frac_volume, e0, eps):
    """
    Calculate effective permittivity using Maxwell-Garnett equation assuming spherical inclusion.

    Note:
        This function is essentially an optimized version of py:func:`maxwell_garnett`.

    Args:
        frac_volume: Fractional volume of snow.
        e0: Permittivity of background (no default, must be provided).
        eps: Permittivity of scattering material (no default, must be provided).

    Returns:
        Effective permittivity.
    """

    Cplus = eps + 2 * e0
    Cminus = (eps - e0) * frac_volume

    Emg = (Cplus + 2 * Cminus) / (Cplus - Cminus) * e0

    return Emg


def _get_depolarization_factors(
    depolarization_factors: Optional[Sequence | Callable],
    inclusion_shape: str,
    frac_volume: float,
    length_ratio: float,
):
    """
    Process the parameter ``depolarizatio_factors`` and depending on its type compute the value of the depolarizationfactors.
    """

    if depolarization_factors is None:
        depol_xyz = depolarization_factors_spheroids(
            inclusion_shape=inclusion_shape,
            frac_volume=frac_volume,
            length_ratio=length_ratio,
        )
    elif callable(depolarization_factors):
        depol_xyz = depolarization_factors(
            inclusion_shape=inclusion_shape,
            frac_volume=frac_volume,
            length_ratio=length_ratio,
        )
    else:
        depol_xyz = depolarization_factors
    return np.asarray(depol_xyz)
