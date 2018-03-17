# coding: utf-8

"""
The helper functions in this module are used to create snowpacks, layers and other media. They are user-friendly and recommended for most usages. Extension of these functions is welcome
on the condition they keep a generic structure. Addition of new, specialized, functions to build more complex snowpacks (e.g. over sea-ice) is also welcome and can be added here.

The function :py:func:`~smrt.inputs.make_medium.make_snowpack` is the first entry point the user should consider to build a snowpack. For example::

    from smrt import make_snowpack

    sp = make_snowpack([1000], density=[300], microstructure_model='sticky_hard_spheres', radius=[0.3e-3], stickiness=0.2)

creates a semi-infinite snowpack made of sticky hard spheres with radius 0.3mm and stickiness 0.2. The :py:obj:`~smrt.core.Snowpack` object is in the `sp` variable.

Note that `make_snowpack` is directly imported from `smrt` instead of `smrt.inputs.make_medium`. This feature is for convenience but is not available
to functions added by the user. These functions must be imported with the full path: ``from smrt.inputs.make_medium import mynewfunction``.


"""


import collections
import numpy as np
import pandas as pd
import six

from smrt.core.snowpack import Snowpack
from smrt.core.globalconstants import FREEZING_POINT, DENSITY_OF_ICE
from smrt.core.layer import get_microstructure_model, Layer
from smrt.core.error import SMRTError
from ..permittivity.saline_ice import brine_permittivity_stogryn85, brine_volume
from ..permittivity.saline_water import seawater_permittivity_klein76


def make_snowpack(thickness, microstructure_model, density,
                  interface=None,
                  substrate=None, **kwargs):

    """
    build a multi-layered snowpack. Each parameter can be an array, list or a constant value.

    :param thickness: thicknesses of the layers in meter (from top to bottom). The last layer thickness can be "numpy.inf" for a semi-infinite layer.
    :param microstructure_model: microstructure_model to use (e.g. sticky_hard_spheres or independent_sphere or exponential)
    :param interface: type of interface, flat/fresnel is the default
    :param density: densities of the layers
    All the other parameters (temperature, microstructure parameters, emmodel, etc, etc) are given as optional arguments (e.g. temperature=[270, 250]).
    They are passed for each layer to the function :py:func:`~smrt.inputs.make_medium.make_snow_layer`. Thus, the documentation of this function is the reference. It describes precisely the available parameters.
    The microstructure parameter(s) depend on the microstructure_model used and is documented in each microstructure_model module.

    TODO: include the documentation of make_snow_layer here once stabilized

    e.g.::

        sp = make_snowpack([1, 10], "exponential", density=[200,300], temperature=[240, 250], corr_length=[0.2e-3, 0.3e-3])

"""

    sp = Snowpack(substrate=substrate)

    if not isinstance(thickness, collections.Iterable):
        raise SMRTError("The thickness argument must be iterable, that is, a list of numbers, numpy array or pandas Series or DataFrame.")


    for i, dz in enumerate(thickness):
        layer = make_snow_layer(dz, get(microstructure_model, i, "microstructure_model"),
                                density=get(density, i, "density"),
                                **get(kwargs, i))

        sp.append(layer, get(interface, i))

    return sp


def make_snow_layer(layer_thickness, microstructure_model,
                    density,
                    temperature=FREEZING_POINT,
                    ice_permittivity_model=None,
                    background_permittivity_model=1,
                    liquid_water=0, salinity=0,
                    **kwargs):

    """Make a snow layer for a given microstructure_model (see also :py:func:`~smrt.inputs.make_medium.make_snowpack` to create many layers).
    The microstructural parameters depend on the microstructural model and should be given as additional arguments to this function. To know which parameters are required or optional,
    refer to the documentation of the specific microstructure model used.

    :param layer_thickness: thickness of snow layer in m
    :param microstructure_model: module name of microstructure model to be used
    :param density: density of snow layer in kg m :sup:`-3`
    :param temperature: temperature of layer in K
    :param permittivity_model: permittivity formulation (default is ice_permittivity_matzler87)
    :param liquid_water: fractional volume of liquid water (default=0)
    :param salinity: salinity in PSU (parts per thousand or g/kg) (default=0)
    :param kwargs: other microstructure parameters are given as optional arguments (in Python words) but may be required (in SMRT words).
    See the documentation of the microstructure model.

    :returns: :py:class:`Layer` instance
"""

    # TODO: Check the validity of the density or see Layer basic_check

    if ice_permittivity_model is None:
        # must import this here instead of the top of the file because of cross-dependencies
        from ..permittivity.ice import ice_permittivity_matzler87  # default ice permittivity model
        ice_permittivity_model = ice_permittivity_matzler87

    # ice in air background. Note that the emmodel might inverse the medium or use other technique for mid-range densities.
    # This is the case of DMRT_Shortrange for instance.
    frac_volume = float(density) / DENSITY_OF_ICE
    eps_1 = background_permittivity_model
    eps_2 = ice_permittivity_model

    if isinstance(microstructure_model, six.string_types):
        microstructure_model = get_microstructure_model(microstructure_model)

    lay = Layer(layer_thickness, microstructure_model,
                frac_volume=frac_volume,
                temperature=temperature,
                permittivity_model=(eps_1, eps_2), **kwargs)

    lay.liquid_water = liquid_water
    lay.salinity = salinity
    lay.density = density # just for information

    return lay


def make_ice_column(thickness, temperature, microstructure_model, inclusion_shape,
                    salinity=0.,
                    brine_volume_fraction=None,
                    add_water_substrate=True,
                    interface=None,
                    substrate=None, **kwargs):
    
    """Build a multi-layered ice column. Each parameter can be an array, list or a constant value.

    :param thickness: thicknesses of the layers in meter (from top to bottom). The last layer thickness can be "numpy.inf" for a semi-infinite layer.
    :param temperature: temperature of ice/water in K
    :param inclusion_shape: assumption for shape of brine inclusions. So far, "spheres" or "random_needles" (i.e. elongated ellipsoidal inclusions) are implemented.
    :param salinity: salinity of ice/water [no units]. Default is 0. If neither salinity nor brine_volume_fraction are given, the ice column is considered to consist of fresh water ice.
    :param brine_volume_fraction: brine / liquid water fraction in sea ice, optional parameter, if not given brine volume fraction is calculated from temperature and salinity in ~.smrt.permittivity.brine_volume_fraction
    :param add_water_substrate: Adds a semi-infinite layer of water below the ice column. Possible arguments are True (default, looks for salinity or brine volume fraction input to determine if a saline or fresh water layer is added), False (no water layer is added), 'ocean' (adds saline water), 'fresh' (adds fresh water layer).
    :param interface: type of interface, flat/fresnel is the default
    All the other parameters (temperature, microstructure parameters, emmodel, etc, etc) are given as optional arguments (e.g. temperature=[270, 250]). Optional arguments are, for example, 'water_temperature', 'water_salinity' and 'water_depth' of the water layer added by 'add_water_substrate'.
    They are passed for each layer to the function :py:func:`~smrt.inputs.make_medium.make_ice_layer`. Thus, the documentation of this function is the reference. It describes precisely the available parameters.

"""

    sp = Snowpack(substrate=substrate)

    for i, dz in enumerate(thickness):
        layer = make_ice_layer(dz, temperature=get(temperature, i),
                               salinity=get(salinity, i),
                               microstructure_model=get(microstructure_model, i),
                               inclusion_shape=get(inclusion_shape, i),
                               brine_volume_fraction=get(brine_volume_fraction, i),
                               **get(kwargs, i))

        sp.append(layer, get(interface, i))

    #add semi-infinite water layer underneath the ice (if wanted):
    water_layer = add_semi_infinite_water_layer(add_water_substrate, salinity, brine_volume_fraction, **kwargs)
    sp.append(water_layer, get(interface, i + 1))
    
    return sp


def make_ice_layer(layer_thickness, temperature, salinity, microstructure_model,
                   inclusion_shape=None,
                   brine_volume_fraction=None,
                   inclusion_permittivity_model=None,
                   background_permittivity_model=None,
                   **kwargs):
    
    """Make an ice layer for a given microstructure_model (see also :py:func:`~smrt.inputs.make_medium.make_ice_column` to create many layers).
    The microstructural parameters depend on the microstructural model and should be given as additional arguments to this function. To know which parameters are required or optional,
    refer to the documentation of the specific microstructure model used.

    :param layer_thickness: thickness of ice layer in m
    :param temperature: temperature of layer in K
    :param salinity: salinity in PSU (parts per thousand or g/kg)
    :param inclusion_shape: assumption for shape of brine inclusions (so far, "spheres" and "random_needles" (i.e. elongated ellipsoidal inclusions) are implemented)
    :param brine_volume_fraction: brine / liquid water fraction in sea ice, optional parameter, if not given brine volume fraction is calculated from temperature and salinity in ~.smrt.permittivity.brine_volume_fraction 
    :param permittivity_model: permittivity formulation (default is ice_permittivity_matzler87)
    :param kwargs: other microstructure parameters are given as optional arguments (in Python words) but may be required (in SMRT words).
    See the documentation of the microstructure model.

    :returns: :py:class:`Layer` instance
"""
    
    if inclusion_permittivity_model is None:
        inclusion_permittivity_model = brine_permittivity_stogryn85 # default brine permittivity model
    if background_permittivity_model is None:
        # 'must import this here instead of the top of the file because of cross-dependencies' is what it says above, so I did the same...
        from ..permittivity.ice import ice_permittivity_matzler87  # default ice permittivity model
        background_permittivity_model = ice_permittivity_matzler87

    eps_1 = background_permittivity_model
    eps_2 = inclusion_permittivity_model

    if brine_volume_fraction is None:
        frac_volume = brine_volume(temperature, salinity)
    else:
        frac_volume = brine_volume_fraction

    if isinstance(microstructure_model, six.string_types):
        microstructure_model = get_microstructure_model(microstructure_model)

    lay = Layer(layer_thickness,
                microstructure_model,
                frac_volume=frac_volume,
                temperature=temperature,
                permittivity_model=(eps_1, eps_2),
                inclusion_shape=inclusion_shape,
                salinity=salinity,
                **kwargs)

    lay.temperature = temperature
    lay.salinity = salinity  # just for information
    lay.frac_volume = frac_volume
    lay.inclusion_shape = inclusion_shape

    return lay


def add_semi_infinite_water_layer(add_water_substrate, salinity, brine_volume_fraction, **kwargs):
    
    """Make a semi-infinite water layer.
    :param add_water_substrate: Possible arguments are True (default, looks for salinity or brine volume fraction input to determine if a saline or fresh water layer is added), False (no water layer is added), 'ocean' (adds saline water), 'fresh' (adds fresh water layer).
    :param salinity: salinity in PSU (parts per thousand or g/kg)
    :param brine_volume_fraction: brine / liquid water fraction in sea ice, optional parameter, if not given brine volume fraction is calculated from temperature and salinity in ~.smrt.permittivity.brine_volume_fraction
    Optional arguments are 'water_temperature', 'water_salinity' and 'water_depth' of the water layer.
    """
    
    if add_water_substrate is True:
        if salinity is None and brine_volume_fraction is None:
            add_water_substrate = "fresh"
        else:
            add_water_substrate = "ocean"

    if add_water_substrate == "ocean":
        water_temperature = FREEZING_POINT - 1.8
        water_salinity = 32. #somewhat arbitrary value, fresher than average ocean salinity, reflecting lower salinities in polar regions
    elif add_water_substrate == "fresh":
        water_temperature = FREEZING_POINT
        water_salinity = 0.
    elif add_water_substrate is not False:
        raise SMRTError("'add_water_substrate' must be set to one of the following: True (default), False, 'ocean', 'fresh'. Additional optional arguments for function make_ice_column are 'water_temperature', 'water_salinity' and 'water_depth'.")

    if add_water_substrate is not False:
        water_depth = 10.  # arbitrary value of 10m thickness for the water layer, microwave absorption in water is usually high, so this represents an infinitely thick water layer

        # override the following variable if set
        water_temperature = kwargs.get('water_temperature', water_temperature)
        water_salinity = kwargs.get('water_salinity', water_salinity)
        water_depth = kwargs.get('water_depth', water_depth)

        inclusion_permittivity_model = seawater_permittivity_klein76 # default sea water permittivity model

        eps_1 = inclusion_permittivity_model
        eps_2 = inclusion_permittivity_model
        
        lay = Layer(water_depth,
                    microstructure_model=get_microstructure_model("exponential"),
                    frac_volume=1.0, # water is considered a uniform medium
                    temperature=water_temperature,
                    permittivity_model=(eps_1, eps_2),
                    salinity=water_salinity,
                    corr_length=0.
                    )

        return lay


def make_generic_stack(thickness, temperature=273, ks=0, ka=0, effective_permittivity=1,
                       interface=None,
                       substrate=None):

    """
    build a multi-layered medium with prescribed scattering and absorption coefficients and effective permittivity. Must be used with presribed_kskaeps emmodel.

    :param thickness: thicknesses of the layers in meter (from top to bottom). The last layer thickness can be "numpy.inf" for a semi-infinite layer.
    :param temperature: temperature of layers in K
    :param ks: scattering coefficient of layers in m^-1
    :param ka: absorption coefficient of layers in m^-1
    :param interface: type of interface, flat/fresnel is the default

"""
# TODO: Add an example
#    e.g.::
#
#        sp = make_snowpack([1, 10], "exponential", density=[200,300], temperature=[240, 250], corr_length=[0.2e-3, 0.3e-3])
#
#"""

    sp = Snowpack(substrate=substrate)

    if not isinstance(thickness, collections.Iterable):
        raise SMRTError("The thickness argument must be iterable, that is, a list of numbers, numpy array or pandas Series or DataFrame.")


    for i, dz in enumerate(thickness):
        layer = make_generic_layer(dz,
                                   ks=get(ks, i, "ks"),
                                   ka=get(ka, i, "ka"),
                                   effective_permittivity=get(effective_permittivity, i, "effective_permittivity"),
                                   temperature=get(temperature, i, "temperature")
                                  )

        sp.append(layer, get(interface, i))

    return sp


def make_generic_layer(layer_thickness, ks=0, ka=0, effective_permittivity=1, temperature=273):
    """Make a generic layer with prescribed scattering and absorption coefficients and effective permittivity. Must be used with presribed_kskaeps emmodel.

    :param layer_thickness: thickness of ice layer in m
    :param temperature: temperature of layer in K
    :param ks: scattering coefficient of layers in m^-1
    :param ka: absorption coefficient of layers in m^-1

    :returns: :py:class:`Layer` instance
"""

    lay = Layer(layer_thickness, frac_volume=1, temperature=temperature)

    lay.temperature = temperature
    lay.effective_permittivity = effective_permittivity
    lay.ks = ks
    lay.ka = ka

    return lay


def get(x, i, name=None):  # function to take the i-eme value in an array or dict of array. Can deal with scalar as well

    if isinstance(x, six.string_types):
        return x
    elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        if i >=len(x.values):
            raise SMRTError("The array '%s' is too short compared to the thickness array" % name)
        return x.values[i]
    if isinstance(x, collections.Sequence) or isinstance(x, np.ndarray):
        if i >=len(x):
            raise SMRTError("The array '%s' is too short compared to the thickness array" % name)
        return x[i]
    elif isinstance(x, dict):
        return {k: get(x[k], i, k) for k in x}
    else:
        return x
