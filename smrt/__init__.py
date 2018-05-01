

# functions to be exported by default

import sys
if sys.version_info[0] == 2:
    import warnings
    warnings.simplefilter('default')
    warnings.warn("SMRT support of Python 2 is going to be dropped soon. Please contact us if this is a major issue for your application", PendingDeprecationWarning)


from .inputs.make_medium import make_snowpack, make_snow_layer, make_ice_column
from .inputs.make_soil import make_soil

from .core.model import make_model, make_emmodel
from .core.error import SMRTError
from .core import sensor
from .core.result import open_result
from .core.sensitivity_study import  sensitivity_study
from .core.globalconstants import PSU, GHz

from .inputs import sensor_list
