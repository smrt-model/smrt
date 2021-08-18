
# coding: utf-8

"""Mixing formulae relevant to snow. This module contains semi-empirical equations to compute the effective permittivity of snow.

Note that by default most emmodels (IBA, DMRT, SFT Rayleigh) uses the generic mixing formula Polder van Staten that mixes the permittivities
of the background (e.g.) and the scatterer materials (e.g. ice) to compute the effective permittivity of snow in a proportion
determined by frac_volume. See py:meth:`~smrt.emmolde.derived_IBA`.

Empirical mixing formulae have been developed for specific mixture of materials (e.g. snow). They can be used to replace 
the Polder van Staten in the EM models. They should not be used to set the material permittivities 
as input of py:meth:`~smrt.smrt_inputs.make_snowpack` and similar functions (because the emmodel would re-mix 
the already mixed materials with the background material).
"""

from ..core.layer import layer_properties
from ..core.globalconstants import FREEZING_POINT, DENSITY_OF_ICE, DENSITY_OF_WATER
from .water import water_permittivity_tiuri80
from .ice import ice_permittivity_tiuri84


@layer_properties("density", "liquid_water")
def wetsnow_permittivity_tiuri80_tinga73(frequency, density, liquid_water):
    """effective permittivity used by Tiuri 1980 based on Tinga and Voss to compute wet snow permittivity

Tiuri, M. and Schultz, H., Theoretical and experimental studies of microwave radiation from a natural snow field. In Rango, A. , ed.
Microwave remote sensing of snowpack properties. Proceedings of a workshop ... Fort Collins, Colorado, May 20-22, 1980. 
Washington, DC, National Aeronautics and Space Center, 225-234. (Conference Publication 2153.)

Tinga, W.R., Voss, W.A.G. and Blossey, D. F.: General approach tomultiphase dielectric mixture theory.
Journal of Applied Physics, Vol.44(1973) No.9,pp.3897-3902.

"""

    # wetness W is the weight percentage of liquid water contained in the snow
    W = liquid_water * DENSITY_OF_WATER / (liquid_water * DENSITY_OF_WATER + (1 - liquid_water) * DENSITY_OF_ICE)

    ri = 0.5e-3  # the result is independent on this value, because only ratio rw/ri or ra/ri or rw/ra are used

    rw = ri * (1 + DENSITY_OF_ICE / DENSITY_OF_WATER * W / (1 - W))**(1 / 3)

    ra = ri * ((DENSITY_OF_ICE / density) * (1 + W / (1 - W)))**(1 / 3)

    eps_a = 1
    eps_w = water_permittivity_tiuri80(frequency, temperature=FREEZING_POINT)
    eps_i = ice_permittivity_tiuri84(frequency, temperature=FREEZING_POINT)

    alpha = 2 * eps_w + eps_i
    beta = eps_w - eps_i

    Es = 1 + 3 * ((rw / ra)**3 * (eps_w - eps_a) * alpha - (ri / ra)**3 * beta * (2 * eps_w + eps_a)) / \
        ((2 + eps_w) * alpha
         - 2 * (ri / rw)**3 * (eps_w - eps_a) * beta
         - (rw / ra)**3 * (eps_w - eps_a) * alpha
         + (ri / ra)**3 * beta * (2 * eps_w + eps_a))

    return Es
