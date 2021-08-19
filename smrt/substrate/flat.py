# coding: utf-8


"""
Implement the flat interface boundary for the bottom layer (substrate). The reflection and transmission
are computed using the Fresnel coefficients. This model does not take any specific parameter.

"""

# local import
from smrt.interface.flat import Flat as iFlat
from smrt.core.interface import substrate_from_interface

# autogenerate from interface.Flat
@substrate_from_interface(iFlat)
class Flat:
    pass
