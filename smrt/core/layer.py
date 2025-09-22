# coding: utf-8

"""
:py:class:`Layer` instance contains all the properties for a single snow layer (e.g. temperature, frac_volume, etc).
It also contains a `microstructure` attribute that holds the microstructural properties (e.g. radius, corr_length, etc).
The class of this attribute defines the microstructure model to use (see :py:mod:`smrt.microstructure_model` package).

To create a single layer, it is recommended to use the function :py:func:`~smrt.inputs.make_medium.make_snow_layer` rather than the class constructor.
However, it is usually more convenient to create a snowpack using :py:func:`~smrt.inputs.make_medium.make_snowpack`.

.. admonition:: **For developers**

    The :py:class:`~smrt.core.layer.Layer` class should not be modified at all even if you need new properties to define the layer (e.g. brine concentration, humidity, ...). If the property you need to add is
    related to geometric aspects, it is probably better to use an existing microstructure model or to create a new one. If the new parameter is not related to geometrical aspect,
    write a function similar to :py:func:`~smrt.inputs.make_medium.make_snow_layer` (choose an explicit name for your purpose). In this function, create the layer by calling the Layer
    constructor as in :py:func:`~smrt.inputs.make_medium.make_snow_layer` and then add your properties with lay.myproperty=xxx, ... See the example of liquid water in :py:func:`~smrt.inputs.make_medium.make_snow_layer`.
    This approach avoids specialization of the Layer class. The new function can be in any file (inc. out of smrt directories), and should be added in :py:mod:`~smrt.inputs.make_medium`
    if it is of general interest and written in a generic way, that is, covers many use cases for many users with default arguments, etc.
"""

import copy
from functools import wraps

# local import
from .error import SMRTError
from .globalconstants import FREEZING_POINT
from .plugin import import_class


class Layer(object):
    """
    Contains the properties for a single layer including the microstructure attribute which holds the microstructure properties.

    To create a layer, it is recommended to use the functions `make_snow_layer` or similar.
    """

    def __init__(
        self,
        thickness,
        microstructure_model=None,
        temperature=FREEZING_POINT,
        permittivity_model=None,
        inclusion_shape=None,
        **kwargs,
    ):
        """
        Builds a snow layer.

        Args:
            thickness (float): Thickness of the snow layer in meters.
            microstructure_model (module, optional): Module name of the microstructure model to be used.
            temperature (float, optional): Temperature of the layer in Kelvin. Defaults to FREEZING_POINT.
            permittivity_model (list or tuple, optional): Permittivity value or model for the background and materials
                (e.g., air and ice). The permittivity can be given as a complex (or real) value or a function that
                returns a value.
            inclusion_shape (str, optional): Assumption for the shape of air/brine inclusions. Options are "spheres",
                "random_needles" (elongated ellipsoidal inclusions), and "mix_spheres_needles".
            **kwargs: Additional parameters for the layer or microstructure model.
        """
        super().__init__()

        self.thickness = thickness

        self.temperature = temperature
        if temperature < 0:
            raise SMRTError("Layer temperature is negative. Temperature must be in Kelvin")

        self.permittivity_model = permittivity_model
        self.inclusion_shape = inclusion_shape

        self.microstructure_model = microstructure_model

        # manage the microstructure parameters
        if microstructure_model is not None:
            valid_args = microstructure_model.valid_arguments()
            microstructure_params = {k: kwargs[k] for k in kwargs if k in valid_args}  # filter valid args

            # make an instance of the micro-structure model
            self.microstructure = microstructure_model(microstructure_params)
        else:
            microstructure_params = {}

        # other params are set in the layer itself
        for k in kwargs:
            if k in microstructure_params:
                continue
            setattr(self, k, kwargs[k])

        self._ssa = getattr(kwargs, "ssa", None)  # save the ssa

    @property
    def ssa(self):
        """
        Returns the SSA, computing it if necessary.
        """
        if not hasattr(self, "_ssa") or self._ssa is None:
            self._ssa = self.microstructure.compute_ssa()
        return self._ssa

    @property
    def frac_volume(self):
        return self.microstructure.frac_volume  # get the frac_volume back from the microstructure

    @frac_volume.setter
    def frac_volume(self, f):
        self.microstructure.frac_volume = f  # set the frac_volume in the microstructure

    def permittivity(self, i, frequency):
        """
        Returns the permittivity of the i-th medium depending on the frequency and internal layer properties.

        Args:
            i (int): Number of the medium. 0 is reserved for the background.
            frequency (float): Frequency of the wave in Hz.

        Returns:
            complex: Permittivity of the i-th medium.

        Raises:
            SMRTError: If the permittivity model is not defined.

        """
        assert i >= 0 and i < len(self.permittivity_model)

        if self.permittivity_model is None:
            raise SMRTError(
                "The permittivity value or model for the background and scatterers must be given when creating a layer"
            )

        if callable(self.permittivity_model[i]):
            # return self.permittivity_model[i](frequency, self.temperature)
            # another approach would be to give the layer object as argument, but this creates a strong dependency
            # between the permittivity and the layer. We prefer to avoid this.
            # Neverthelees, if the list of arguments should lengthen, it will be better to pass the object.
            # A elegant/functional alternative would be to use function of the temperature only, and use "functools.partial"
            # when creating the layer to include the dependency to the layer properties.

            # the chosen approach is based on the decorator required_layer_properties. See below. It allows the permittivity model to
            # be called with the layer argument, but the initial permittivity_model function never heard about layers
            return self.permittivity_model[i](frequency, layer_to_inject=self)

        else:  # assume it is independent of the frequency.
            return self.permittivity_model[i]

    def basic_checks(self):
        """
        Performs basic input checks on the layer information.

        Checks:
            - Temperature is between 100 and the freezing point (Kelvin units check).
            - Density is between 1 and DENSITY_OF_ICE (SI units check).
            - Layer thickness is above zero.


        Raises:
            SMRTError: If any of the checks fail.
        """
        if self.layer_thickness <= 0:
            raise SMRTError("Layer thickness must be positive")

        if self.microstructure_model is None:
            raise SMRTError("No microstructure_model has been defined")

        if hasattr(self, "temperature"):
            if self.temperature < 100:
                raise SMRTError("Temperature should be in Kelvin, got %g" % self.temperature)

            if self.temperature > FREEZING_POINT:
                # This warning should not be applied to substrates i.e. soil and open water
                raise SMRTError("Temperature is above freezing, got %g" % self.temperature)

        if self.frac_volume < 0 or self.frac_volume > 1:
            raise SMRTError("Check density units are kg per m3")

    def inverted_medium(self):
        """
        Returns the layer with inverted autocorrelation and inverted permittivities.

        Returns:
            Layer: A new layer object with inverted properties.

        Raises:
            SMRTError: If the microstructure model does not support inversion.
        """
        obj = copy.deepcopy(self)
        if not hasattr(self.microstructure, "inverted_medium"):
            raise SMRTError("The microstructure model does not support model inversion")

        obj.microstructure = self.microstructure.inverted_medium()
        obj.permittivity_model = tuple(reversed(self.permittivity_model))
        return obj

    def __setattr__(self, name, value):
        if hasattr(self, "read_only_attributes") and (name in self.read_only_attributes):
            raise SMRTError(
                f"The attribute '{name}' is read-only, because setting its value requires recalculation."
                " In general, this is solved by using the update method."
            )

        super().__setattr__(name, value)

        # the callback system has been deactivated and replaced by the update method.
        # See here for why the option __setattr__ has been chosen:
        # https://stackoverflow.com/questions/17576009/python-class-property-use-setter-but-evade-getter
        # if hasattr(self, "attributes_with_callback") and (name in self.attributes_with_callback):
        #     # get the callback function
        #     callback = self.attributes_with_callback[name]
        #     # call the callback function
        #     callback(self, name)

    def update(self, **kwargs):
        """
        Updates the attributes. This method is to be used when recalculation of the state of the object
        is necessary. See for instance :py:class:`~smrt.inputs.make_medium.SnowLayer`.

        Args:
            **kwargs:
        """
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_microstructure_model(modulename, classname=None):
    """
    Returns the class corresponding to the microstructure_model defined in modulename.

    This function imports the correct module if possible and returns the class.
    It is used internally and should not be needed for normal usage.

    Args:
        modulename: name of the python module in smrt/microstructure_model
        classname:  (Default value = None)
    """
    # import the module
    return import_class("microstructure_model", modulename)


def make_microstructure_model(modelname_or_class, **kwargs):
    """
    Creates a microstructure instance.

    Args:
        modelname_or_class (str or type): Name of the module or directly the class.
        **kwargs: Arguments needed for the specific autocorrelation.

    Args:
        modelname_or_class:
        **kwargs:

    Returns:
        instance of the autocorrelation `modelname` with the parameters given in `**kwargs`

    :Example:

         To import the StickyHardSpheres class with spheres radius of 1mm, stickiness of 0.5 and fractional_volume of 0.3::

         shs = make_autocorrelation("StickyHardSpheres", radius=0.001, stickiness=0.5, frac_volume=0.3)
    """

    if isinstance(modelname_or_class, str):
        cls = get_microstructure_model(modelname_or_class)
    else:
        cls = modelname_or_class

    return cls(kwargs)  # sent as an array as need by the constructor.


def layer_properties(*required_arguments, optional_arguments=None, **kwargs):
    """
    This decorator is used for the permittivity functions (or any other functions) to inject layer's attributes as arguments.
    The decorator declares the layer properties needed to call the function and the optional ones.
    This allows permittivity functions to use any property of the layer, as long as it is defined.

    Args:
        *required_arguments:
        optional_arguments:  (Default value = None)
        **kwargs:
    """

    def wrapper(f):
        @wraps(f)
        def newf(*args, layer_to_inject=None, **kwargs):
            if layer_to_inject is not None:
                args = list(args)  # make it mutable
                assert isinstance(layer_to_inject, Layer)  # this is not stricly required

                for ra in required_arguments:
                    if hasattr(layer_to_inject, ra):
                        kwargs[ra] = getattr(
                            layer_to_inject, ra
                        )  # add the layer's attributes as named arguments (avoid problems)
                    else:
                        raise Exception(
                            "The layer must have the '%s' attribute to call the function %s " % (ra, str(f))
                        )
                if optional_arguments:
                    for ra in optional_arguments:
                        if hasattr(layer_to_inject, ra):
                            kwargs[ra] = getattr(
                                layer_to_inject, ra
                            )  # add the layer's over the eventual default arguments
            return f(*args, **kwargs)

        return newf

    return wrapper
