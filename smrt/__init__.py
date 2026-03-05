# functions to be exported by default

import sys

if (sys.version_info[0] == 2) or ((sys.version_info[0] == 3) and (sys.version_info[1] <= 9)):
    raise RuntimeError("Python 3.9 and lower are not supported anymore")

# if (sys.version_info[0] == 3) and (sys.version_info[1] <= 9):
#     from warnings import warn
#     warn('Support of Python 3.9 will be discarded in 2025 to implement type hinting', DeprecationWarning)


from .core import sensor as sensor
from .core.error import SMRTError as SMRTError
from .core.globalconstants import (
    PSU as PSU,
)
from .core.globalconstants import (
    GHz as GHz,
)
from .core.globalconstants import (
    cm as cm,
)
from .core.globalconstants import (
    micron as micron,
)
from .core.globalconstants import (
    mm as mm,
)
from .core.model import (
    make_emmodel as make_emmodel,
)
from .core.model import (
    make_model as make_model,
)
from .core.model import (
    make_rtsolver as make_rtsolver,
)
from .core.plugin import register_package as register_package
from .core.result import open_result as open_result
from .core.sensitivity_study import sensitivity_study as sensitivity_study
from .inputs import sensor_list as sensor_list
from .inputs.make_medium import (
    make_atmosphere as make_atmosphere,
)
from .inputs.make_medium import (
    make_ice_column as make_ice_column,
)
from .inputs.make_medium import (
    make_interface as make_interface,
)
from .inputs.make_medium import (
    make_snow_layer as make_snow_layer,
)
from .inputs.make_medium import (
    make_snowpack as make_snowpack,
)
from .inputs.make_medium import (
    make_water_body as make_water_body,
)
from .inputs.make_soil import make_soil as make_soil
from .inputs.make_soil import make_soil_substrate as make_soil_substrate
