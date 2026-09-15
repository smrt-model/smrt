"""Deprecated compatibility module.

Use ``smrt.inputs.altimeter_list`` instead.
"""

from warnings import warn

from smrt.inputs.altimeter_list import asiras_lam as asiras_lam
from smrt.inputs.altimeter_list import cryosat2 as cryosat2
from smrt.inputs.altimeter_list import envisat_ra2 as envisat_ra2
from smrt.inputs.altimeter_list import sentinel3_sral as sentinel3_sral

warn("The module lrm_altimeter_list is deprecated and will be removed in a future version.", DeprecationWarning)
