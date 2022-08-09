# coding: utf-8

""" :py:class:`Layer` instance contains all the properties for a single snow layer (e.g. temperature, frac_volume, etc).
It also contains a `microstructure` attribute that holds the microstructural properties (e.g. radius, corr_length, etc).
The class of this attribute defines the microstructure model to use (see :py:mod:`smrt.microstructure_model` package).

To create a single layer, it is recommended to use the function :py:func:`~smrt.inputs.make_medium.make_snow_layer` rather than the class constructor. However it is usually more convenient
to create a snowpack using :py:func:`~smrt.inputs.make_medium.make_snowpack`.

.. admonition:: **For developers**

    The :py:class:`~smrt.core.layer.Layer` class should not be modified at all even if you need new properties to define the layer (e.g. brine concentration, humidity, ...). If the property you need to add is
    related to geometric aspects, it is probably better to use an existing microstructure model or to create a new one. If the new parameter is not related to geometrical aspect,
    write a function similar to :py:func:`~smrt.inputs.make_medium.make_snow_layer` (choose an explicit name for your purpose). In this function, create the layer by calling the Layer
    constructor as in :py:func:`~smrt.inputs.make_medium.make_snow_layer` and then add your properties with lay.myproperty=xxx, ... See the example of liquid water in :py:func:`~smrt.inputs.make_medium.make_snow_layer`.
    This approach avoids specialization of the Layer class. The new function can be in any file (inc. out of smrt directories), and should be added in :py:mod:`~smrt.inputs.make_medium`
    if it is of general interest and written in a generic way, that is, covers many use cases for many users with default arguments, etc.

"""

from functools import wraps
import copy

# local import
from .error import SMRTError
from .plugin import import_class

from .globalconstants import FREEZING_POINT


class Layer(object):
    """ Contains the properties for a single layer including the microstructure attribute which holds the microstructure properties.

    To create layer, it is recommended to use of the functions :py:meth:`make_snow_layer` and similar

    """

    def __init__(self, thickness, microstructure_model=None,
                 temperature=FREEZING_POINT, permittivity_model=None, inclusion_shape=None,
                 **kwargs):
        """ Build a snow layer.

        :param thickness: thickness of snow layer in m
        :param microstructure_model: module name of microstructure model to be used
        :param temperature: temperature of layer in K
        :param permittivity_model: list or tuple of permittivity value or model for the background and materials (e.g. air and ice). The permittivity can be
        given as a complex (or real) value or a function that return a value (see :py:mod:`smrt.permittivity` modules)
        :param inclusion_shape: assumption for shape of air/brine inclusions (so far, "spheres" and "random_needles" (i.e. elongated ellipsoidal inclusions) and "mix_spheres_needles" are implemented)

"""
        super().__init__()

        self.thickness = thickness
        # TODO Ghi: send a warning for non valid_args
        if thickness == 0:
            raise SMRTError("Layer with thickness = 0 (or even <~wavelength) is not recommended, part of the code does not support it.")

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

        # other params
        for k in kwargs:
            if k in microstructure_params:
                continue
            setattr(self, k, kwargs[k])

        self._ssa = getattr(kwargs, 'ssa', None)  # save the ssa

    @property
    def ssa(self):
        """return the SSA, compute it if necessary"""

        if not hasattr(self, '_ssa') or self._ssa is None:
            self._ssa = self.microstructure.compute_ssa()
        return self._ssa

    @property
    def frac_volume(self):
        return self.microstructure.frac_volume  # get the frac_volume back from the microstructure

    @frac_volume.setter
    def frac_volume(self, f):
        self.microstructure.frac_volume = f  # set the frac_volume in the microstructure

    def permittivity(self, i, frequency):
        """return the permittivity of the i-th medium depending on the frequency and internal layer properties. Usually i=0 is air and i=1 is ice for dry snow with a low or moderate density.

    :param i: number of the medium. 0 is reserved for the background
    :param frequency: frequency of the wave (Hz)

    :returns: complex permittivity of the i-th medium
"""

        assert i >= 0 and i < len(self.permittivity_model)

        if self.permittivity_model is None:
            raise SMRTError("The permittivity value or model for the background and scatterers must be given when creating a layer")

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
        """ Function to provide very basic input checks on the layer information

        Currently checks:

        * temperature is between 100 and the freezing point (Kelvin units check),
        * density is between 1 and DENSITY_OF_ICE (SI units check)
        * layer thickness is above zero

        """

        if self.layer_thickness <= 0:
            raise SMRTError("Layer thickness must be positive")

        if self.microstructure_model is None:
            raise SMRTError("No microstructure_model has been defined")

        if hasattr(self, "temperature"):
            if self.temperature < 100:
                raise SMRTError('Temperature should be in Kelvin, got %g' % self.temperature)

            if self.temperature > FREEZING_POINT:
                # This warning should not be applied to substrates i.e. soil and open water
                raise SMRTError('Temperature is above freezing, got %g' % self.temperature)

        if self.frac_volume < 0 or self.frac_volume > 1:
            raise SMRTError('Check density units are kg per m3')

    def inverted_medium(self):
        """return the layer with inverted autocorrelation and inverted permittivities.
        """

        obj = copy.deepcopy(self)
        if not hasattr(self.microstructure, "inverted_medium"):
            raise SMRTError("The microstructure model does not support model inversion")

        obj.microstructure = self.microstructure.inverted_medium()
        obj.permittivity_model = tuple(reversed(self.permittivity_model))
        return obj

    def __setattr__(self, name, value):

        if hasattr(self, "read_only_attributes") and (name in self.read_only_attributes):
            raise SMRTError(f"The attribute '{name}' is read-only, because setting its value requires recalculation."
                            " In general, this is solved by using the update method.")

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
        """update the attributes. This method is to be used when recalculation of the state of the object
        is necessary. See for instance :py:class:`~smrt.inputs.make_medium.SnowLayer`.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_microstructure_model(modulename, classname=None):
    """return the class corresponding to the microstructure_model defined in modulename.

    This function import the correct module if possible and return the class.
    It is used internally and should not be needed for normal usage.

    :param modulename: name of the python module in smrt/microstructure_model
    """

    # import the module
    return import_class("microstructure_model", modulename)


def make_microstructure_model(modelname_or_class, **kwargs):
    """create an microstructure instance.

    This function is called internally and should not be needed for normal use.

    :param modelname_or_class: name of the module or directly the class.
    :param type: string

    :param **kwargs: all the arguments need for the specific autocorrelation.

    :returns: instance of the autocorrelation `modelname` with the parameters given in `**kwargs`

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
    """This decorator is used for the permittivity functions (or any other functions) to inject layer's attributes as arguments.
The decorator declares the layer properties needed to call the function and the optional ones.
This allows permittivity functions to use any property of the layer, as long as it is defined. """

    def wrapper(f):
        @wraps(f)
        def newf(*args, layer_to_inject=None, **kwargs):
            if layer_to_inject is not None:
                args = list(args)  # make it mutable
                assert isinstance(layer_to_inject, Layer)  # this is not stricly required

                for ra in required_arguments:
                    if hasattr(layer_to_inject, ra):
                        kwargs[ra] = getattr(layer_to_inject, ra)  # add the layer's attributes as named arguments (avoid problems)
                    else:
                        raise Exception("The layer must have the '%s' attribute to call the function %s " % (ra, str(f)))
                if optional_arguments:
                    for ra in optional_arguments:
                        if hasattr(layer_to_inject, ra):
                            kwargs[ra] = getattr(layer_to_inject, ra)  # add the layer's over the eventual default arguments
            return f(*args, **kwargs)
        return newf
    return wrapper
