

# functions to be exported by default

import sys
if sys.version_info[0] == 2:
    raise RuntimeError("Pyhton 2.7 is not supported anymore")


from .inputs.make_medium import make_snowpack, make_snow_layer, make_ice_column, make_atmosphere, make_interface, make_water_body
from .inputs.make_soil import make_soil

from .core.model import make_model, make_emmodel
from .core.error import SMRTError
from .core import sensor
from .core.result import open_result
from .core.sensitivity_study import  sensitivity_study
from .core.globalconstants import PSU, GHz, cm, mm, micron
from .core.plugin import register_package

from .inputs import sensor_list
