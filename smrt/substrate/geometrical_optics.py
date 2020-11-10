# coding: utf-8


"""
Implement the geometrical optics rough substrate. See the documentation in :py:mod:`~smrt.interface.geometrical_optics`

"""

# local import
from smrt.interface.geometrical_optics import GeometricalOptics as iGeometricalOptics
from smrt.core.interface import substrate_from_interface

# autogenerate from interface.GeometricalOptics
@substrate_from_interface(iGeometricalOptics)
class GeometricalOptics:
    pass
