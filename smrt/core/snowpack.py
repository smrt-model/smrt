
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
import numpy as np
import warnings
import functools

from .error import SMRTError
from ..interface.flat import Flat  # core should not depend on something defined in interface...
from .layer import Layer
from .interface import SubstrateBase

cached_property = getattr(functools, "cached_property", property)  # use cached_propertity if it exists... otherwise, use property


class Snowpack(object):
    """holds the description of the snowpack, including the layers, interfaces, and the substrate

"""

    def __init__(self, layers=None, interfaces=None, substrate=None, atmosphere=None):
        self.layers = layers if layers is not None else list()
        self.update_layer_number()

        self.interfaces = interfaces if interfaces is not None else list()
        self.substrate = substrate
        self.atmosphere = atmosphere  # this is temporary as in the future atmosphere will/may be normal layers

    @property
    def nlayer(self):
        """return the number of layers
"""
        return len(self.layers)

    @cached_property
    def layer_thicknesses(self):
        """return the thickness of each layer
"""
        return [lay.thickness for lay in self.layers]  # TODO Ghi: caching

    @property
    def layer_depths(self):
        """return the depth of the bottom of each layer

"""
        warnings.warn("layer_depths is ambiguous, use bottom_layer_depths, top_layer_depths or mid_layer_depths instead."
                      "This function will be removed in a next version",
                      DeprecationWarning)
        return np.cumsum(self.layer_thicknesses)

    @cached_property
    def bottom_layer_depths(self):
        """return the depth of the bottom of each layer

"""
        return np.cumsum(self.profile('thickness'))  # TODO Ghi: caching

    @property
    def top_layer_depths(self):
        """return the depth of the bottom of each layer

"""
        return self.z[:-1]

    @cached_property
    def mid_layer_depths(self):
        """return the depth of the bottom of each layer

"""
        ld = self.z
        return (ld[1:] + ld[:-1]) / 2

    @cached_property
    def z(self):
        """return the depth of each interface, that is, 0 and the depth of the bottom of each layer

"""
        return np.insert(self.layer_depths, 0, 0)

    @property
    def layer_densities(self):
        """return the density of each layer
"""
        warnings.warn("layer_densities is ambiguous, use the profile('density') instead. This function will be removed in a next version",
                      DeprecationWarning)
        return [lay.density for lay in self.layers]  # TODO Ghi: caching

    # @functools.lru_cache()  # this has side effect when layers are changed after calling this function
    def profile(self, property_name):
        """return the property of each layer as a list
"""
        if property_name == "bottom_layer_depths":
            return self.bottom_layer_depths
        elif property_name == "top_layer_depths":
            return self.top_layer_depths
        elif property_name == "mid_layer_depths":
            return self.mid_layer_depths
        else:
            return [getattr(lay, property_name) for lay in self.layers]

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
        changing the original snowpack. See py:meth:~deepcopy.
"""
        new_sp = copy.copy(self)
        new_sp.layers = copy.copy(self.layers)
        new_sp.interfaces = copy.copy(self.interfaces)
        return new_sp

    def deepcopy(self):
        """make a deep copy of a snowpack.
"""
        return copy.deepcopy(self)

    def basic_check(self):

        if len(self.interfaces) != len(self.layers):
            raise SMRTError("The number of layers must equal the number of interfaces")

    def check_addition_validity(self, other):

        # import here to avoid circular reference
        from .atmosphere import AtmosphereBase

        if isinstance(other, Layer):
            # this is valid, pass
            pass
        elif isinstance(other, SubstrateBase):
            if self.substrate is not None:
                raise SMRTError("Adding a substrate to a snowpack that already has a substrate set is not valid."
                                " Unset the substrate first.")
        elif isinstance(other, AtmosphereBase):
            raise SMRTError("Adding an atmosphere to a snowpack is not allowed. Add an atmosphere and a snowpack.")
        elif not (hasattr(other, "layers") and hasattr(other, "interfaces") and hasattr(other, "substrate")):
            raise SMRTError("Addition of snowpacks requires two instances of class Snowpack or equivalent compatible objects")

        elif self.substrate is not None:
            raise SMRTError("While adding snowpacks, the first (topmost) snowpack must not have a substrate. Unset the substrate"
                            " before adding the two snowpacks.")
        elif other.atmosphere is not None:
            raise SMRTError("While adding snowpacks, the second (bottommost) snowpack must not have an atmosphere. Unset the atmosphere"
                            " before adding the two snowpacks.")

    def update_layer_number(self):

        for i in range(len(self.layers)):
            self.layers[i].number = i

    def __add__(self, other):
        """Return a new snowpack made of the first snowpack (or layer) stacked on top of the second snowpack (or layer or substrate).

        .. note:: if a layer is added on top (at bottom), the top (bottom) interface is duplicated.

        :param other: the snowpack, a layer or a substrate to add to the first argument.

        :Example:

        # duplicate the top layer:    
        newsp = sp.layers[0] + wetsp

"""
        self.check_addition_validity(other)

        if isinstance(other, SubstrateBase):
            return Snowpack(layers=self.layers,
                            interfaces=self.interfaces,
                            atmosphere=self.atmosphere,
                            substrate=other)
        elif isinstance(other, Layer):
            newsp = copy.deepcopy(self)
            newsp += copy.deepcopy(other)
            return newsp
        else:
            return Snowpack(layers=self.layers + other.layers,
                            interfaces=self.interfaces + other.interfaces,
                            substrate=other.substrate,
                            atmosphere=self.atmosphere)

    def __radd__(self, other):

        if other is 0:
            return self
        elif isinstance(other, Layer):

            newsp = copy.deepcopy(self)
            newsp.layers.insert(0, copy.deepcopy(other))
            newsp.interfaces.insert(0, copy.deepcopy(self.interfaces[0]))  # duplicate the upper interface
            newsp.update_layer_number()
            return newsp
        else:
            # should never be called
            raise SMRTError("The addition operator is not commutative for snowpacks")
            return other.__add__(self)

    def __iadd__(self, other):  # just for optimization
        """Inplace addition of object to snowpack. See :func:`~snowpack.Snowpack.__add__` description.

"""
        self.check_addition_validity(other)

        if isinstance(other, SubstrateBase):
            self.substrate = other
        elif isinstance(other, Layer):
            self.layers.append(copy.deepcopy(other))
            self.interfaces.append(copy.deepcopy(self.interfaces[-1]))  # duplicate the bottomost layer
            self.update_layer_number()
        else:
            self.layers += other.layers
            self.interfaces += other.interfaces
            self.substrate = other.substrate
            self.update_layer_number()

        return self
