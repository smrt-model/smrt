"""
   This modules provides the :py:class:`Terrain` class. Instances of this class contains the topography of the terrain,
   a list of snowpacks and a map associating each cell of the terrain to the snowpack list.

   It is in development and the API is subject to change.

Example::

    # create a 10-m thick snowpack with a single layer,
    # density is 350 kg/m3. The exponential autocorrelation function is
    # used to describe the snow and the "size" parameter is therefore
    # the correlation length which is given as an optional
    # argument of this function (but is required in practice)

    sp = make_snowpack([10], "exponential", [350], corr_length=[3e-3])


"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

from smrt.core.error import SMRTError
from smrt.core.globalconstants import LOG2
from smrt.core.snowpack import Snowpack


@dataclass
class TerrainInfo(object):
    """Hold information about the topography. Can be a DEM or a parameterized random distribution with slope. This class
    can be used as an interface or a proper dataclass.
    """

    distribution: str = "normal"

    sigma_surface: float = 0.0
    slope_angle: float = 0.0
    slope_direction: float = 0.0

    dem: Optional[npt.NDArray[np.float64]] = None


@dataclass
class Terrain(object):
    snowpacks: list[Snowpack]
    terrain_info: TerrainInfo
    snowpack_map: npt.NDArray[np.integer]


random_generator = None


def generate_dem(terrain_info: TerrainInfo, x, y):
    """generate a DEM according to the distribution and sigma_surface in terrain_info. Currently the terrain correlation
    is not considered."""

    global random_generator

    if random_generator is None:
        random_generator = default_rng()

    # get the nx and ny size
    if (len(x.shape) == 1) and (len(y.shape) == 1):
        nx, ny = len(x), len(y)
    elif (len(x.shape) == 2) and (x.shape[0] == 1) and (len(y.shape) == 2) and (y.shape[1] == 1):
        nx, ny = x.size, y.size
    else:
        raise SMRTError("x and y does not have a adequate structure")

    if terrain_info.dem is None:
        if terrain_info.sigma_surface > 0:
            if terrain_info.distribution == "normal":
                dem = random_generator.normal(scale=terrain_info.sigma_surface, size=(ny, nx))
            elif terrain_info.distribution == "lognormal":
                mu = -LOG2 / 2  # np.log(1 / np.sqrt(1 + h2)) with h2=1  # based on Landy et al. 2019 (CODE)
                var = LOG2  # np.log(1 + h2)
                dem = random_generator.lognormal(mean=mu, sigma=np.sqrt(var), size=(ny, nx))
                dem -= np.mean(dem)
                dem *= terrain_info.sigma_surface / np.std(dem)
            else:
                raise SMRTError("Unknown terrain topography distribution")
        else:
            dem = np.zeros((ny, nx))
    else:
        dem = terrain_info.dem
        assert dem.shape[0] == ny
        assert dem.shape[1] == nx

    return dem
