from dataclasses import dataclass

import numpy as np
import xarray as xr

from .error import SMRTError
from .snowpack import Snowpack

# this is a temporary solution to deal with Atmosphere.
# user should not rely on this object. For temporary internal use only !!


class AtmosphereBase(object):
    # has no special properties yet, we just use the type.

    def __add__(self, other):
        """
        Return a new snowpack made by setting the atmosphere.

        Args:
            other: The snowpack to add.

        Returns:
            Snowpack: A new snowpack with the atmosphere set.

        Raises:
            SMRTError: If the other object is not a Snowpack.
        """

        if not isinstance(other, Snowpack):
            raise SMRTError(
                "Attempt to add an incorrect object to an atmopshere. Only adding an atmosphere and a snowpack (in that order)"
                " is a valid operation."
            )

        return Snowpack(
            layers=other.layers,
            interfaces=other.interfaces,
            substrate=other.substrate,
            terrain_info=other.terrain_info,
            atmosphere=self,
        )

    def __iadd__(self, other):
        raise SMRTError("Inplace addition with an atmosphere is not a valid operation.")


@dataclass
class AtmosphereResult:
    tb_down: np.ndarray
    tb_up: np.ndarray
    transmittance: np.ndarray
    coords: dict = None

    def save(self, filename, netcdf_engine=None):
        ds = xr.Dataset(
            {
                "tb_down": (self.coords, self.tb_down),
                "tb_up": (self.coords, self.tb_up),
                "transmittance": (self.coords, self.transmittance),
            },
            coords=self.coords,
        )
        ds.to_netcdf(filename, engine=netcdf_engine)
