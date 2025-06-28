# coding: utf-8

"""
Implements the geometrical optics rough substrate.

See the documentation in smrt.interface.geometrical_optics_backscatter.
"""

# local import
from smrt.interface.transparent import Transparent as iTransparent
from smrt.core.interface import substrate_from_interface

# autogenerate from interface.GeometricalOptics
@substrate_from_interface(iTransparent)
class Transparent:
    pass


