"""
:py:class:`Snowpack` instance contains the description of the snowpack, including a list of layers and interfaces between the layers, and the substrate (soil, ice, ...).

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
import pandas as pd
import warnings
import functools

from .error import SMRTError
from ..interface.flat import Flat  # core should not depend on something defined in interface...
from .layer import Layer
from .interface import SubstrateBase


class Snowpack(object):
    """
    Holds the description of the snowpack, including the layers, interfaces, and the substrate.
    """

    def __init__(self, layers=None, interfaces=None, substrate=None, atmosphere=None, terrain_info=None):
        super().__init__()

        self.layers = layers if layers is not None else list()  # list of the layers
        self.update_layer_number()

        self.interfaces = interfaces if interfaces is not None else list()  # list of the interface
        self.substrate = substrate  # substrate at the bottom of the snowpack
        self.atmosphere = atmosphere  # this is temporary as in the future atmosphere will/may be normal layers
        self.terrain_info = terrain_info  # provide information about the terrain

    @property
    def nlayer(self):
        """
        Returns the number of layers.
        """
        return len(self.layers)

    @property
    def thickness(self):
        return sum(lay.thickness for lay in self.layers)

    @property
    def layer_thicknesses(self):
        """
        Returns the thickness of each layer.
        """
        return [lay.thickness for lay in self.layers]  # TODO Ghi: caching

    @property
    def layer_depths(self):
        """
        Returns the depth of the bottom of each layer.
        """
        warnings.warn(
            "layer_depths is ambiguous, use bottom_layer_depths, top_layer_depths or mid_layer_depths instead."
            "This function will be removed in a next version",
            DeprecationWarning,
        )
        return np.cumsum(self.layer_thicknesses)

    @property
    def bottom_layer_depths(self):
        """
        Returns the depth of the bottom of each layer.
        """
        return np.cumsum(self.profile("thickness"))  # TODO Ghi: caching

    @property
    def top_layer_depths(self):
        """
        Returns the depth of the bottom of each layer.
        """
        return self.z[:-1]

    @property
    def mid_layer_depths(self):
        """
        Returns the depth of the bottom of each layer.
        """
        ld = self.z
        return (ld[1:] + ld[:-1]) / 2

    @property
    def z(self):
        """
        Returns the depth of each interface, that is, 0 and the depths of the bottom of each layer.
        """
        return np.insert(self.bottom_layer_depths, 0, 0)

    @property
    def layer_densities(self):
        """
        Returns the density of each layer.
        """
        warnings.warn(
            "layer_densities is ambiguous, use the profile('density') instead. This function will be removed in a next version",
            DeprecationWarning,
        )
        return [lay.density for lay in self.layers]  # TODO Ghi: caching

    def profile(self, property_name, where="all", raise_attributeerror=False):
        """
        Returns the vertical profile of property_name. The property is searched either in the layer, microstructure or interface.

        Args:
            property_name (str): Name of the property.
            where (str): Where to search the property. Can be 'all', 'layer', 'microstructure', or 'interface'.
            raise_attributeerror (bool): Raise an attribute error if the attribute is not found.

        Returns:
            np.ndarray: Array of the property values.
        """
        if property_name == "bottom_layer_depths":
            return self.bottom_layer_depths
        elif property_name == "top_layer_depths":
            return self.top_layer_depths
        elif property_name == "mid_layer_depths":
            return self.mid_layer_depths

        if where == "layer":
            prof = [getattr(lay, property_name, None) for lay in self.layers]
        elif where == "microstructure":
            prof = [getattr(lay.microstructure, property_name, None) for lay in self.layers]
        elif where == "interface":
            prof = [getattr(i, property_name, None) for i in self.interfaces]
        elif where == "all":
            assert len(self.layers) == len(self.interfaces)
            prof = [
                getattr(self.layers[i], property_name)
                if hasattr(self.layers[i], property_name)
                else getattr(self.layers[i].microstructure, property_name)
                if hasattr(self.layers[i].microstructure, property_name)
                else getattr(self.interfaces[i], property_name, None)
                for i in range(len(self.layers))
            ]
        else:
            raise ValueError("invalid value for 'where' argument")

        if raise_attributeerror and all((p is None for p in prof)):
            raise AttributeError("The attribute %s can not be found" % property_name)

        return np.array(prof)

    def append(self, layer, interface=None):
        """
        Appends a new layer at the bottom of the stack of layers. The interface is that at the top of the appended layer.

        Args:
            layer (Layer): Instance of Layer.
            interface: Type of interface. By default, flat surface (Flat) is considered meaning the coefficients are calculated with Fresnel coefficient
                and using the effective permittivity of the surrounding layers.
        """

        if not isinstance(layer, Layer):
            raise Warning(
                "the layer to append in the snowpack is not an instance of the class Layer. This may be a mistake in your code."
            )

        layer.number = 0 if not self.layers else self.layers[-1].number + 1
        self.layers.append(layer)

        # interface can be a class or an instance. Both must work equaly.
        if interface is None:
            interface = Flat()
        self.interfaces.append(interface)

    def delete(self, ilayer):
        """
        Deletes a layer and the upper interface.

        Args:
            ilayer (int): Index of the layer.
        """

        warnings.warn(
            "The delete method will be depreciated in the future. Use the delete_layer method instead which is exactly equivalent.",
            DeprecationWarning,
        )
        return self.delete_layer(ilayer)

    def delete_layer(self, ilayer):
        """
        Deletes a layer and the upper interface.

        Args:
            ilayer (int): Index of the layer.
        """
        self.layers.pop(ilayer)
        self.interfaces.pop(ilayer)

    def delete_bottom(self, ilayer):
        """
        Deletes the bottom of the snowpack from layer n. Deletes also the substrate.

        Args:
            ilayer (int): Index of the first layer to delete.
        """
        assert ilayer < self.nlayer
        self.layers = self.layers[0:ilayer]
        self.interfaces = self.layers[0:ilayer]
        self.substrate = None

    def shallow_copy(self, cut_bottom=None):
        """
        Make a shallow copy of a snowpack by copying the list of layers and interfaces but not the layers and
        interfaces themselves which are still shared with the original snowpack.

        This method allows the advanced user to create a new snowpack and remove, append or replace some layers or
        interfaces afterward. It does not allow to alter the layers or interfaces without
        changing the original snowpack. See deep_copy.

        .. warning::
           This function is for advanced users, understanding the concept and consequence of shallow copy.
           It is likely to generate bugs. It may be removed in the future, if snowpack becomes immutable.

        Args:
            cut_bottom (int, optional): If cut_bottom is a number, all layers below the layer indexed by 'cut_bottom' are removed as well as the substrate.

        Returns:
            Snowpack: The shallow copy of the snowpack.
        """

        new_sp = copy.copy(self)

        if (cut_bottom is None) or (cut_bottom >= self.nlayer):
            cut_bottom = self.nlayer
        else:
            new_sp.substrate = None

        new_sp.layers = copy.copy(self.layers[0:cut_bottom])
        new_sp.interfaces = copy.copy(self.interfaces[0:cut_bottom])
        return new_sp

    def copy(self, cut_bottom=None):
        warnings.warn(
            "The copy method will be depreciated in the future. Use the shallow_copy method instead which is exactly equivalent.",
            DeprecationWarning,
        )
        return self.shallow_copy(cut_bottom=cut_bottom)

    def deepcopy(self):
        """
        Makes a deep copy of a snowpack.

        Returns:
            Snowpack: The deep copy of the snowpack.
        """
        warnings.warn(
            "The deepcopy method will be depreciated in the future. Use the deep_copy method instead which is exactly equivalent.",
            DeprecationWarning,
        )
        return copy.deepcopy(self)

    def deep_copy(self):
        """
        Makes a deep copy of a snowpack.

        Returns:
            Snowpack: The deep copy of the snowpack.
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
                raise SMRTError(
                    "Adding a substrate to a snowpack that already has a substrate set is not valid."
                    " Unset the substrate first."
                )
        elif isinstance(other, AtmosphereBase):
            raise SMRTError("Adding an atmosphere to a snowpack is not allowed. Add an atmosphere and a snowpack.")
        elif not (hasattr(other, "layers") and hasattr(other, "interfaces") and hasattr(other, "substrate")):
            raise SMRTError(
                "Addition of snowpacks requires two instances of class Snowpack or equivalent compatible objects"
            )

        elif self.substrate is not None:
            raise SMRTError(
                "While adding snowpacks, the first (topmost) snowpack must not have a substrate. Unset the substrate"
                " before adding the two snowpacks."
            )
        elif other.atmosphere is not None:
            raise SMRTError(
                "While adding snowpacks, the second (bottommost) snowpack must not have an atmosphere. Unset the atmosphere"
                " before adding the two snowpacks."
            )

    def update_layer_number(self):
        for i in range(len(self.layers)):
            self.layers[i].number = i

    def __add__(self, other):
        """
        Returns a new snowpack made of the first snowpack (or layer) stacked on top of the second snowpack (or layer or substrate).

        .. note:: if a layer is added on top (at bottom), the top (bottom) interface is duplicated.

        Args:
        other: the snowpack, a layer or a substrate to add to the first argument.

        :Example:

        # duplicate the top layer:
        newsp = sp.layers[0] + wetsp
        """
        if other == 0:
            return self

        self.check_addition_validity(other)

        if isinstance(other, SubstrateBase):
            return Snowpack(layers=self.layers, interfaces=self.interfaces, atmosphere=self.atmosphere, substrate=other)
        elif isinstance(other, Layer):
            newsp = copy.deepcopy(self)
            newsp += copy.deepcopy(other)
            return newsp
        else:
            return Snowpack(
                layers=self.layers + other.layers,
                interfaces=self.interfaces + other.interfaces,
                substrate=other.substrate,
                atmosphere=self.atmosphere,
            )

    def __radd__(self, other):
        if other == 0:
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
        """
        Inplace addition of object to snowpack. See :func:`~snowpack.Snowpack.__add__` description.
        """
        self.check_addition_validity(other)

        if other == 0:
            pass
        elif isinstance(other, SubstrateBase):
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

    def to_dataframe(self, default_columns=True, other_columns=None):
        columns = [
            "thickness",
            "microstructure_model",
            "density",
            "temperature",
            "liquid_water",
            "salinity",
            "ice_type",
        ]

        def multi_index(index1, index2):
            return {(index1, i2): 1 for i2 in index2}  # use order in dict (Python >3.7) as a ordered set

        columns = multi_index("layer", columns)

        # add microstructure parameters
        for lay in self.layers:
            columns.update(multi_index("microstructure", lay.microstructure.args))
            columns.update(multi_index("microstructure", lay.microstructure.optional_args))

        # add interface parameters
        columns.update({("interface", "name"): 1})
        for i in self.interfaces:
            columns.update(multi_index("interface", i.args))
            columns.update(multi_index("interface", i.optional_args))

        df = pd.DataFrame()

        # add layer attribute
        for c in columns:
            if c == ("interface", "name"):
                df[c] = [type(i) for i in self.interfaces]
            else:
                df[c] = self.profile(c[1], where=c[0])

        if self.substrate is not None:
            substrate = {("substrate", "name"): type(self.substrate)}
            for args in self.substrate.args:
                substrate[("substrate", args)] = [getattr(self.substrate, args)]
            for args in self.substrate.optional_args:
                substrate[("substrate", args)] = [getattr(self.substrate, args)]

            df = pd.concat((df, pd.DataFrame(substrate, index=["s"])))  # append
            # reorder
            df = df[list(columns.keys()) + list(substrate.keys())]

        # convert class in class name
        df = df.applymap(lambda x: x.__name__ if isinstance(x, type) else x)
        # df = df.map(lambda x: x.__name__ if isinstance(x, type) else x)  # for pandas >2.1.0

        # use multi index
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        # remove na column

        return df.dropna(axis=1, how="all")

    def __repr__(self):
        return "Snowpack: " + repr(self.to_dataframe())

    def _repr_html_(self):
        """
        Used by IPython notebook to display a snowpack in a pretty format
        """

        return "Snowpack: " + self.to_dataframe().to_html(notebook=True, na_rep="--", justify="start")
