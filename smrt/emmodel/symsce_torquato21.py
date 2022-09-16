# coding: utf-8

"""Compute scattering with the symmetrized  version of the Strong-Contrast Expansion (SCE) from Torquato and Kom 2021 
under the non-local approximation, a.k.a long range in Tsang's books. The truncation of the series is at second order.
"""

# Stdlib import

# other import

# local import
from smrt.permittivity.generic_mixing_formula import polder_van_santen
from .common import AdjustableEffectivePermittivityMixins, derived_EMModel
from .sce_common import SCEBase


#
# For developers: all emmodel must implement the `effective_permittivity`, `ke` and `phase` functions with the same arguments as here
# initialisation and precomputation can be done in the prepare method that is called only once for each layer whereas
# phase, ke and effective_permittivity can be called several times.
#


def derived_SymSCETK21(effective_permittivity_model):
    """return a new SymSCE model with variant from the default SymSCE.

    :param effective_permittivity_model: permittivity mixing formula.

    :returns a new class inheriting from SymSCE but with patched methods
    """

    return derived_EMModel(SymSCETK21, effective_permittivity_model)


class SymSCETK21(AdjustableEffectivePermittivityMixins, SCEBase):

    effective_permittivity_model = staticmethod(polder_van_santen)

    def __init__(self, sensor, layer, scaled=True):

        super().__init__(sensor, layer, local=False, symmetrical=True, scaled=scaled)
