"""Implement the iiem rough substrate. See the documentation in :py:mod:`~smrt.interface.iiem_fung02`"""

# local import
from smrt.core.interface import substrate_from_interface
from smrt.interface.iiem_fung02 import IIEM_Fung02 as iIIEM_Fung02


# autogenerate from interface.GeometricalOptics
@substrate_from_interface(iIIEM_Fung02)
class IIEM_Fung02:
    pass
