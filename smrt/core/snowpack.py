
""":py:class:`Snowpack` instance contains the description of the snowpack, including a list of layers and interfaces between the layers, and the substrate (soil, ice, ...).

To create a snowpack, it is recommended to use the :py:func:`~smrt.inputs.make_medium.make_snowpack` function which avoids the complexity of creating
each layer and then the snowpack from the layers. For more complex media (like lake ice or sea ice), it may be necessary to directly call the functions
to create the different layers (such as :py:func:`~smrt.inputs.make_medium.make_snow_layer`).

Example::

    # create a 10-m thick snowpack with a single layer,
    # density is 350 kg/m3. The exponential autocorrelation function is
    # used to describe the snow and the "size" parameter is therefore
    # the correlation length which is given as an optional
    # argument of this function (but is required in practice)

    sp = make_snowpack([10], "exponential", [350], corr_length=[3e-3])


"""

import collections
import numpy as np
import pandas as pd
import six

from .error import SMRTError
from ..interface.flat import Flat





class Snowpack(object):
    """holds the description of the snowpack, including the layers, interfaces, and the substrate

"""

    def __init__(self, layers=None, interfaces=None, substrate=None):
        self.layers = layers if layers is not None else list()
        self.interfaces = interfaces if interfaces is not None else list()
        self.substrate = substrate

    @property
    def nlayer(self):
        """return the number of layers
"""
        return len(self.layers)

    @property
    def layer_thicknesses(self):
        """return the thickness of each layer
"""
        return [lay.thickness for lay in self.layers]  # TODO Ghi: caching

    @property
    def layer_depths(self):
        """return the depth of the bottom of each layer

"""
        return np.cumsum(self.thicknesses)  # TODO Ghi: caching

    def append(self, layer, interface=None):
        """append a new layer at the bottom of the stack of layers. The interface is that at the top of the appended layer.

    :param layer: instance of :py:class:`~layer.Layer`
    :param interface: type of interface. By default, flat surface (:py:class:`~..interface.flat.Flat`) is considered meaning the coefficients are calculated with Fresnel coefficient
                      and using the effective permittivity of the surrounding layers
"""

        self.layers.append(layer)

        # interface can be a class or an instance. Both must work equaly.
        if interface is None:
            interface = Flat
        self.interfaces.append(interface)

    def basic_check(self):

        if len(self.interfaces) != len(self.layers):
            raise SMRTError("The number of layers must equal the number of interfaces")
