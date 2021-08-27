
# coding: utf-8

"""Mixing formulae relevant to snow. This module contains equations to compute the effective permittivity of snow.

Note that by default most emmodels (IBA, DMRT, SFT Rayleigh) uses the generic mixing formula Polder van Staten that mixes the permittivities
of the background (e.g.) and the scatterer materials (e.g. ice) to compute the effective permittivity of snow in a proportion
determined by frac_volume. See py:meth:`~smrt.emmolde.derived_IBA`.

Many semi-empirical mixing formulae have been developed for specific mixture of materials (e.g. snow). They can be used to replace
the Polder van Staten in the EM models. They should not be used to set the material permittivities
as input of py:meth:`~smrt.smrt_inputs.make_snowpack` and similar functions (because the emmodel would re-mix
the already mixed materials with the background material).
"""

import numpy as np
from ..core.layer import layer_properties
from ..core.globalconstants import FREEZING_POINT, DENSITY_OF_ICE, DENSITY_OF_WATER


@layer_properties("density", "liquid_water", optional_arguments=["ice_permittivity_model", "water_permittivity_model"])
def wetsnow_permittivity_tinga73(frequency, density, liquid_water, ice_permittivity_model=None, water_permittivity_model=None):
    """effective permittivity proposed by Tinga et al. 1973 for three-component mixing. The component 1 is the background ("a" here),
    the compoment 2 ("w" here) is a spherical shell surrounding the component 3 ("i" here).

     It was used by Tiuri as well as T. Mote to compute wet snow permittivity.

Tinga, W.R., Voss, W.A.G. and Blossey, D. F.: General approach to multiphase dielectric mixture theory.
Journal of Applied Physics, Vol.44(1973) No.9,pp.3897-3902.
doi: /10.1063/1.1662868


Tiuri, M. and Schultz, H., Theoretical and experimental studies of microwave radiation from a natural snow field. In Rango, A. , ed.
Microwave remote sensing of snowpack properties. Proceedings of a workshop ... Fort Collins, Colorado, May 20-22, 1980.
Washington, DC, National Aeronautics and Space Center, 225-234. (Conference Publication 2153.)


"""

    # wetness W is the weight percentage of liquid water contained in the snow
    W = liquid_water * DENSITY_OF_WATER / (liquid_water * DENSITY_OF_WATER + (1 - liquid_water) * DENSITY_OF_ICE)

    # equation for spheres. Here we rather defined V to avoid the exponentiation
    # ri = 0.5e-3  # the result is independent on this value, because only ratio rw/ri or ra/ri or rw/ra are used

    # rw = ri * (1 + DENSITY_OF_ICE / DENSITY_OF_WATER * W / (1 - W))**(1 / 3)

    # ra = ri * ((DENSITY_OF_ICE / density) * (1 + W / (1 - W)))**(1 / 3)

    Vw_i = 1 + DENSITY_OF_ICE / DENSITY_OF_WATER * W / (1 - W)
    Va_i = (DENSITY_OF_ICE / density) * (1 + W / (1 - W))

    if water_permittivity_model is None:
        from .water import water_permittivity_tiuri80
        water_permittivity_model = water_permittivity_tiuri80
    if ice_permittivity_model is None:
        from .ice import ice_permittivity_tiuri84
        ice_permittivity_model = ice_permittivity_tiuri84

    eps_a = 1
    eps_w = water_permittivity_model(frequency, temperature=FREEZING_POINT)
    eps_i = ice_permittivity_model(frequency, temperature=FREEZING_POINT)  # this must be dry ice !

    alpha = 2 * eps_w + eps_i
    diff_wi = eps_w - eps_i
    diff_wa = eps_w - eps_a

    denominator = (2 * eps_a + eps_w) * alpha - 2 * (1 / Vw_i) * diff_wa * diff_wi \
        - (Vw_i / Va_i) * diff_wa * alpha \
        + (1 / Va_i) * diff_wi * (2 * eps_w + eps_a)

    Es = eps_a * (1 + 3 * ((Vw_i / Va_i) * diff_wa * alpha - (1 / Va_i) * diff_wi * (2 * eps_w + eps_a)) / denominator)

    # t is possible to compute the square_field_ratio_tinga73
    # using np.abs(eps_w * eps_a / denominator)**2
    # to be implemented

    return Es


@layer_properties("density")
def depolarization_factors_maetzler96(density):
    """The empirical depolarization factors of snow estimated by Mäzler 1996. It is supposed to provide more accurate permittivity=f(density)
     than using constant depolarization factors in Polder van Santen (e.g. spheres)

Biblio: C. Mäzler, Microwave Permittivity of dry snow, IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 34, NO. 2, MARCH 1996
"""
    frac_volume = density / DENSITY_OF_ICE   # this way to compute frac_volume avoid inversion of the medium

    if frac_volume < 0.33:
        A = 0.1 + 0.5 * frac_volume
    elif frac_volume < 0.71:
        A = 0.18 + 3.24 * (frac_volume - 0.49)**2
    else:
        A = 1 / 3
    return np.array([A, A, 1 - 2 * A])


@layer_properties("density")
def drysnow_permittivity_maetzler96(density, e0=1, eps=3.185):

    if (e0.real > 1) and (eps == 1):
        e0, eps = eps, e0

    assert e0.real < eps.real

    frac_volume = density / DENSITY_OF_ICE   # this way to compute frac_volume avoid inversion of the medium

    A = depolarization_factors_maetzler96(density)

    # A = np.array([1 / 3., 1 / 3., 1 / 3.]) # Spheres. For testing

    eps_diff = eps - e0

    # Solve Polder van Santen with an iterative approach (could be optimized)
    # rough first guess
    eps_eff0 = frac_volume * eps + (1 - frac_volume) * e0

    for i in range(20):  # use an inefficient iterative approach
        eps_app = e0 * A + eps_eff0 * (1 - A)

        eps_eff = e0 + frac_volume * eps_diff * np.sum(eps_app / (eps_app + A * eps_diff)) \
            / (3 - frac_volume * eps_diff * np.sum(A / (eps_app + A * eps_diff)))

        if np.abs(eps_eff - eps_eff0) < 1e-6:
            break
        eps_eff0 = eps_eff  # new estimate becomes first guess

    # last estimation of eps_app: eps_app = e0 + (1 - A) * eps_eff

    return eps_eff
