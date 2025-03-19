# functions to be exported by default

import sys

if (sys.version_info[0] == 2) or ((sys.version_info[0] == 3) and (sys.version_info[1] <= 8)):
    raise RuntimeError('Python 3.8 and lower are not supported anymore')

if (sys.version_info[0] == 3) and (sys.version_info[1] <= 9):
    from warnings import warn
    warn('Support of Python 3.9 will be discarded in 2025 to implement type hinting')


from .inputs.make_medium import (
    make_snowpack,
    make_snow_layer,
    make_ice_column,
    make_atmosphere,
    make_interface,
    make_water_body,
)

from .inputs.make_soil import make_soil

from .core.model import make_model, make_emmodel, make_rtsolver
from .core.error import SMRTError
from .core import sensor
from .core.result import open_result
from .core.sensitivity_study import sensitivity_study
from .core.globalconstants import PSU, GHz, cm, mm, micron
from .core.plugin import register_package

from .inputs import sensor_list
