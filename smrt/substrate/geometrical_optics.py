# coding: utf-8

"""
Implements the geometrical optics rough substrate.

See the documentation in smrt.interface.geometrical_optics.
"""

# local import
from smrt.core.interface import substrate_from_interface
from smrt.interface.geometrical_optics import GeometricalOptics as iGeometricalOptics


# autogenerate from interface.GeometricalOptics
@substrate_from_interface(iGeometricalOptics)
class GeometricalOptics:
    pass
