# coding: utf-8


"""
Implement the flat interface boundary for the bottom layer (substrate). The reflection and transmission
are computed using the Fresnel coefficients. This model does not take any specific parameter.

"""

# local import
from smrt.interface.iem_fung92_brogioni10 import IEM_Fung92_Briogoni10 as iIEM_Fung92_Briogoni10
from smrt.core.interface import substrate_from_interface

# autogenerate from interface.Flat
@substrate_from_interface(iIEM_Fung92_Briogoni10)
class IEM_Fung92_Briogoni10:
    pass
