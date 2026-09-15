"""Deprecated compatibility module.

Use ``smrt.inputs.altimeter_list`` instead.
"""

from warnings import warn

from smrt.inputs.altimeter_list import cryosat2_sarm as cryosat2_sarm
from smrt.inputs.altimeter_list import sentinel3_sarm as sentinel3_sarm

warn("The module lrm_altimeter_list is deprecated and will be removed in a future version.", DeprecationWarning)
