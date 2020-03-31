
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

import copy
import collections
import numpy as np
import pandas as pd
import inspect

from .error import SMRTError
from ..interface.flat import Flat  # core should not depend on something defined in interface...
from .layer import Layer
from .plugin import import_class
from .interface import SubstrateBase


class Snowpack(object):
    """holds the description of the snowpack, including the layers, interfaces, and the substrate

"""

    def __init__(self, layers=None, interfaces=None, substrate=None):
        self.layers = layers if layers is not None else list()
        self.update_layer_number()

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
        return np.cumsum(self.layer_thicknesses)  # TODO Ghi: caching

    @property
    def z(self):
        """return the depth of each interface, that is, 0 and the depth of the bottom of each layer

"""
        return np.insert(self.layer_depths, 0, 0)

    def append(self, layer, interface=None):
        """append a new layer at the bottom of the stack of layers. The interface is that at the top of the appended layer.

    :param layer: instance of :py:class:`~layer.Layer`
    :param interface: type of interface. By default, flat surface (:py:class:`~..interface.flat.Flat`) is considered meaning the coefficients are calculated with Fresnel coefficient
                      and using the effective permittivity of the surrounding layers
"""

        if not isinstance(layer, Layer):
            raise Warning("the layer to append in the snowpack is not an instance of the class Layer. This may be a mistake in your code.")

        layer.number = 0 if not self.layers else self.layers[-1].number + 1
        self.layers.append(layer)

        # interface can be a class or an instance. Both must work equaly.
        if interface is None:
            interface = Flat()
        self.interfaces.append(interface)

    def delete(self, ilayer):
        """delete a layer and the upper interface

    :param ilayer: index of the layer
"""
        self.layers.pop(ilayer)
        self.interfaces.pop(ilayer)

    def copy(self):
        """make a shallow copy of a snowpack by copying the list of layers and interfaces but not the layers and interfaces themselves which are still shared with the original snowpacl.
        This method allows the user to create a new snowpack and remove, append or replace some layers or interfaces afterward. It does not allow to alter the layers or interfaces without 
        changing the original snowpack. In this latter case, use copy.deepcopy.
"""
        new_sp = copy.copy(self)
        new_sp.layers = copy.copy(self.layers)
        new_sp.interfaces = copy.copy(self.interfaces)
        return new_sp

    def basic_check(self):

        if len(self.interfaces) != len(self.layers):
            raise SMRTError("The number of layers must equal the number of interfaces")

    def check_addition_validity(self, other):

        if isinstance(other, SubstrateBase):
            if self.substrate is not None:
                raise SMRTError("Adding a substrate to a snowpack that already has one is not valid. Unset the substrate first")            
        elif not (hasattr(other, "layers") and hasattr(other, "interfaces") and hasattr(other, "substrate")):
            raise SMRTError("Addition of snowpacks requires two instances of class Snowpack or equivalent compatible objects")

        elif self.substrate is not None:
            raise SMRTError("While adding snowpacks, the first (topmost) snowpack must not have a substrate. Unset the substrate"
                            " before adding the two snowpacks.")

    def update_layer_number(self):

        for i in range(len(self.layers)):
            self.layers[i].number = i

    def __add__(self, other):
        """Return a new snowpack made of the first snowpack stacked on top of the second one.

        :param other: the snowpack or substrate to add to the first argument."""

        self.check_addition_validity(other)

        if isinstance(other, SubstrateBase):
            return Snowpack(layers=self.layers,
                            interfaces=self.interfaces,
                            substrate=other)
        else:
            return Snowpack(layers=self.layers + other.layers,
                            interfaces=self.interfaces + other.interfaces,
                            substrate=other.substrate)

    def __radd__(self, other):

        if other is 0:
            return self
        else:
            # should never be called
            return other.__add__(self)

    def __iadd__(self, other):  # just for optimization
        """Inplace addition of object to snowpack. See :func:`~snowpack.Snowpack.__add__` description.

"""
        self.check_addition_validity(other)

        if isinstance(other, SubstrateBase):
            self.substrate = other
        else:
            self.layers += other.layers
            self.interfaces += other.interfaces
            self.substrate = other.substrate
            self.update_layer_number()

        return self
