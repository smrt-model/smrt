from dataclasses import dataclass

import numpy as np
import xarray as xr

from smrt.core.lib import planck_function

from .error import SMRTError
from .snowpack import Snowpack

# this is a temporary solution to deal with Atmosphere.
# user should not rely on this object. For temporary internal use only !!


class AtmosphereBase(object):
    # has no special properties yet, we just use the type.

    def run(self, frequency, costheta, npol, rayleigh_jeans_approximation=False):
        raise NotImplementedError("The run method must be implemented in subclasses.")

    def __add__(self, other):
        """Return a new snowpack made by setting the atmosphere.

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

    def run(self, frequency, costheta, npol, rayleigh_jeans_approximation=False):
        res0 = self.atmospheres[0].run(
            frequency, costheta, npol, rayleigh_jeans_approximation=rayleigh_jeans_approximation
        )

        if len(self.atmospheres) == 1:
            return res0

        for atmos in self.atmospheres[1:]:
            res = atmos.run(frequency, costheta, npol, rayleigh_jeans_approximation=rayleigh_jeans_approximation)

            res0.intensity_down = res0.intensity_down * res.transmittance + res.intensity_down
            res0.intensity_up = res0.intensity_up + res0.transmittance * res.intensity_up
            res0.transmittance = res0.transmittance * res.transmittance

        return res0

    def __iadd__(self, other):
        self.atmospheres.append(other)
        return self


@dataclass
class AtmosphereResult:
    intensity_down: np.ndarray
    intensity_up: np.ndarray
    transmittance: np.ndarray
    coords: dict = None
    rayleigh_jeans_approximation: bool = False

    def save(self, filename, netcdf_engine=None):
        ds = {
            "transmittance": (self.coords, self.transmittance),
            "intensity_down": (self.coords, self.intensity_down),
            "intensity_up": (self.coords, self.intensity_up),
        }

        ds = xr.Dataset(ds, coords=self.coords)
        ds.to_netcdf(filename, engine=netcdf_engine)

    @property
    def tb_down(self):
        if not self.rayleigh_jeans_approximation:
            raise SMRTError("tb_down is only available when rayleigh_jeans_approximation is True.")
        return self.intensity_down

    @property
    def tb_up(self):
        if not self.rayleigh_jeans_approximation:
            raise SMRTError("tb_up is only available when rayleigh_jeans_approximation is True.")
        return self.intensity_up


def make_nonscattering_atmosphere_results(
    frequency, tb_down, tb_up, transmittance, coords=None, rayleigh_jeans_approximation=False
):
    """Make an AtmosphereResult for a non-scattering atmosphere given the downwelling and upwelling brightness temperatures
    and the transmittance. The nonscattering assumption is used to compute the intensity from the brightness temperature
    by calculating the emissivity and using the Planck function. The Rayleigh-Jeans approximation can be used to
    directly use the brightness temperature as intensity.

    Args:
        frequency: Frequency in Hz.
        tb_down: Downwelling brightness temperature in K.
        tb_up: Upwelling brightness temperature in K.
        transmittance: Transmittance of the atmosphere (between 0 and 1).
        coords (dict, optional): Coordinates for the output AtmosphereResult. Defaults to None.
        rayleigh_jeans_approximation (bool, optional): Whether to use the Rayleigh-Jeans approximation. Defaults to False.
    """
    if rayleigh_jeans_approximation:
        intensity_down = tb_down
        intensity_up = tb_up
    else:
        e = (1 - transmittance).clip(1e-5, 1)
        intensity_down = planck_function(frequency, tb_down / e) * e
        intensity_up = planck_function(frequency, tb_up / e) * e

    return AtmosphereResult(
        intensity_down=intensity_down,
        intensity_up=intensity_up,
        transmittance=transmittance,
        coords=coords,
        rayleigh_jeans_approximation=rayleigh_jeans_approximation,
    )
