# coding: utf-8

"""Compute scattering with the the Strong-Contrast Expansion (SCE) from Torquato and Kom 2021 
This SCE is the "non-local approximation" in Torquato also called "long range" in Tsang's books
It applies to scatterer size up to 1 wavelength.
"""

# Stdlib import

# other import

# local import
from smrt.permittivity.generic_mixing_formula import maxwell_garnett_for_spheres
from .common import AdjustableEffectivePermittivityMixins, derived_EMModel
from .sce_common import SCEBase

#
# For developers: all emmodel must implement the `effective_permittivity`, `ke` and `phase` functions with the same arguments as here
# initialisation and precomputation can be done in the prepare method that is called only once for each layer whereas
# phase, ke and effective_permittivity can be called several times.
#


def derived_SCETK21(effective_permittivity_model):
    """return a new SCE_ShortRange model with variant from the default SCE_ShortRange.

    :param effective_permittivity_model: permittivity mixing formula.

    :returns a new class inheriting from SCE_ShortRange but with patched methods
    """

    return derived_EMModel(SCETK21, effective_permittivity_model)


class SCETK21(AdjustableEffectivePermittivityMixins, SCEBase):

    """
        To be documented
    """

    # default effective_permittivity_model is maxwell_garnett according to the SCE theory
    effective_permittivity_model = staticmethod(maxwell_garnett_for_spheres)

    def __init__(self, sensor, layer, scaled=True):

        super().__init__(sensor, layer, local=False, symmetrical=False, scaled=scaled)
