

from .error import SMRTError
from .snowpack import Snowpack


# this is a temporary solution to deal with Atmosphere.
# user should not rely on this object. For temporary internal use only !!

class AtmosphereBase(object):
    # has no special properties yet, we just use the type.

    def __add__(self, other):
        """Return a new snowpack made by setting the atmosphere

        :param other: the snowpack to add.
"""

        if not isinstance(other, Snowpack):
            raise SMRTError("Attempt to add an incorrect object to an atmopsher. Only adding an atmosphere and a snowpack (in that order)"
                            " is a valid operation.")

        return Snowpack(layers=other.layers,
                        interfaces=other.interfaces,
                        substrate=other.substrate,
                        atmosphere=self)

    def __iadd__(self, other):
        raise SMRTError("Inplace addition with an atmosphere is not a valid operation.")
