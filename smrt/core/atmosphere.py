from dataclasses import dataclass

import numpy as np
import xarray as xr

from .error import SMRTError
from .snowpack import Snowpack

# this is a temporary solution to deal with Atmosphere.
# user should not rely on this object. For temporary internal use only !!


class AtmosphereBase(object):
    # has no special properties yet, we just use the type.

    def run(self, frequency, costheta, npol):
        raise NotImplementedError("The run method must be implemented in subclasses.")

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

        match other:
            case AtmosphereBase():
                # adding two atmospheres results in stacking them
                return AtmosphereStack([self, other])

            case Snowpack():
                # adding an atmosphere to a snowpack results in setting the atmosphere or adding a stack of atmospheres

                if other.atmosphere is not None:
                    # adding to an existing atmosphere results in stacking
                    new_atmosphere = AtmosphereStack([self, other.atmosphere])
                else:
                    new_atmosphere = self

                return Snowpack(
                    layers=other.layers,
                    interfaces=other.interfaces,
                    substrate=other.substrate,
                    terrain_info=other.terrain_info,
                    atmosphere=new_atmosphere,
                )

            case _:
                raise SMRTError(
                    "Attempt to add an incorrect object to an atmopshere. Only adding an atmosphere and a snowpack (in that order)"
                    " is a valid operation."
                )

    def __iadd__(self, other):
        raise SMRTError("Inplace addition with an atmosphere is not a valid operation.")


class AtmosphereStack(AtmosphereBase):
    """Atmosphere stack made by stacking multiple atmosphere objects in series.
    The atmospheres are stacked from top to bottom.
    """

    def __init__(self, atmospheres):
        """Stack multiple atmosphere objects in series.

        Args:
            atmospheres (list): List of atmospheres to stack, from top to bottom.
        """
        self.atmospheres = atmospheres

    def run(self, frequency, costheta, npol):
        res0 = self.atmospheres[0].run(frequency, costheta, npol)

        if len(self.atmospheres) == 1:
            return res0

        for atmos in self.atmospheres[1:]:
            res = atmos.run(frequency, costheta, npol)

            res0.tb_down = res0.tb_down * res.transmittance + res.tb_down
            res0.tb_up = res0.tb_up + res0.transmittance * res.tb_up
            res0.transmittance = res0.transmittance * res.transmittance

        return res0

    def __iadd__(self, other):
        self.atmospheres.append(other)
        return self


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
