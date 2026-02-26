# coding: utf-8
"""
Computes scattering with the symmetrized version of the Strong-Contrast Expansion (SCE) from Torquato and Kom 2021 under
the non-local approximation, a.k.a long range in Tsang's books. The truncation of the series is at second order.

References:
    - Torquato, S., & Kim, J. (2021). Nonlocal Effective Electromagnetic Wave Characteristics of
         Composite Media: Beyond the Quasistatic Regime. Physical Review X, 11(2). doi:10.1103/physrevx.11.021002
    - G. Picard, H. Löwe, C. Mätzler, A continuous formulation of microwave scattering from fresh snow to bubbly ice
         from first principles, The Cryosphere, 16, 3861–3866,doi:10.5194/tc-16-3861-2022, 2022
"""

# Stdlib import

# other import

# local import
from smrt.permittivity.generic_mixing_formula import polder_van_santen

from .common import AdjustableEffectivePermittivityMixin, derived_EMModel
from .sce_common import SCEBase

#
# For developers: all emmodel must implement the `effective_permittivity`, `ke` and `phase` functions with the same arguments as here
# initialisation and precomputation can be done in the prepare method that is called only once for each layer whereas
# phase, ke and effective_permittivity can be called several times.
#


def derived_SymSCETK21(effective_permittivity_model):
    """
    Returns a new SymSCE model with variant from the default SymSCE.

    Args:
        effective_permittivity_model: Permittivity mixing formula.

    Returns:
        class: A new class inheriting from SymSCE but with patched methods.
    """

    return derived_EMModel(SymSCETK21, effective_permittivity_model)


class SymSCETK21(AdjustableEffectivePermittivityMixin, SCEBase):
    effective_permittivity_model = staticmethod(polder_van_santen)

    def __init__(self, sensor, layer, scaled=True):
        super().__init__(sensor, layer, local=False, symmetrical=True, scaled=scaled)
