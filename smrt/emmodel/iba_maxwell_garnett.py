# coding: utf-8

"""Compute scattering from Improved Born Approximation theory with altered effective and apparent permittivities. 

This version is a modification of the original model to use the Maxwell-Garnet (MG) permittivity instead of the Polder
van Santen (PvS) as the effective permittivity. For consistency, the apparent permittivity used in IBA (Mätzler, 1998)
is also adapted based on Shivola and Kong, 1988 which details the relationship between apparent and effective
permititviies and gives examples for MG and PvS.

This is modification is useful for instance to compare with the Strong Constrast Expansion (the non symmetrized
versions) which uses the MG permittivity.
"""

# Stdlib import

# other import

import numpy as np

from smrt.permittivity.generic_mixing_formula import maxwell_garnett
from .iba import IBA


class IBA_MaxewellGarnett(IBA):

    """
    Modified Improved Born Approximation electromagnetic model class.

    As with all electromagnetic modules, this class is used to create an electromagnetic
    object that holds information about the effective permittivity, extinction coefficient and
    phase function for a particular snow layer. Due to the frequency dependence, information
    about the sensor is required. Passive and active sensors also have different requirements on
    the size of the phase matrix as redundant information is not calculated for the
    passive case.

    :param sensor: object containing sensor characteristics
    :param layer: object containing snow layer characteristics (single layer)

    """
    effective_permittivity_model = staticmethod(maxwell_garnett)


    def mean_sq_field_ratio(self, e0, eps):
        """ Mean squared field ratio calculation

        Uses layer effective permittivity

        :param e0: background relative permittivity
        :param eps: scattering constituent relative permittivity

        """
        apparent_permittivity = e0
        y2 = (1. / 3.) * np.sum(np.absolute(apparent_permittivity / (apparent_permittivity + (eps - e0) * self.depol_xyz))**2.)
        return y2
