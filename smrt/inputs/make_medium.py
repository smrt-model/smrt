# coding: utf-8

"""
The helper functions in this module are used to create snowpacks, sea-ice and other media. They are user-friendly and recommended for most usages. Extension of these functions is welcome
on the condition they keep a generic structure.

The function :py:func:`~smrt.inputs.make_medium.make_snowpack` is the first entry point the user should consider to build a snowpack. For example::

    from smrt import make_snowpack

    sp = make_snowpack([1000], density=[300], microstructure_model='sticky_hard_spheres', radius=[0.3e-3], stickiness=0.2)

creates a semi-infinite snowpack made of sticky hard spheres with radius 0.3mm and stickiness 0.2. The :py:obj:`~smrt.core.Snowpack` object is in the `sp` variable.

Note that `make_snowpack` is directly imported from `smrt` instead of `smrt.inputs.make_medium`. This feature is for convenience.

"""


import collections
from functools import partial

import numpy as np
import pandas as pd
import six

from smrt.core.snowpack import Snowpack
from smrt.core.globalconstants import FREEZING_POINT, DENSITY_OF_ICE, PERMITTIVITY_OF_AIR
from smrt.core.layer import get_microstructure_model, Layer
from smrt.core.error import SMRTError
from smrt.permittivity.ice import ice_permittivity_maetzler06  # default pure ice permittivity model
from smrt.permittivity.brine import brine_volume
from smrt.permittivity.saline_water import seawater_permittivity_klein76, brine_permittivity_stogryn85
from smrt.permittivity.saline_ice import saline_ice_permittivity_pvs_mixing

from smrt.substrate.flat import Flat

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
    :param salinity: salinity in kg/kg, for using PSU as unit see PSU constant in smrt module (default = 0)
    :param kwargs: other microstructure parameters are given as optional arguments (in Python words) but may be required (in SMRT words).
    See the documentation of the microstructure model.

    :returns: :py:class:`Layer` instance
"""

    # TODO: Check the validity of the density or see Layer basic_check

    if ice_permittivity_model is None:
        # must import this here instead of the top of the file because of cross-dependencies
        from ..permittivity.ice import ice_permittivity_maetzler06  # default ice permittivity model
        ice_permittivity_model = ice_permittivity_maetzler06

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


def make_ice_column(ice_type,
                    thickness, temperature, microstructure_model,
                    brine_inclusion_shape='spheres',
                    salinity=0.,
                    brine_volume_fraction=None,
                    brine_permittivity_model=None,
                    ice_permittivity_model=None,
                    saline_ice_permittivity_model=None,
                    porosity=0.,
                    density=None,
                    add_water_substrate=True,
                    interface=None,
                    substrate=None, **kwargs):
    """Build a multi-layered ice column. Each parameter can be an array, list or a constant value.

   ice_type variable determines the type of ice, which has a big impact on how the medium is modelled and the parameters:
    - First year is modelled as scattering brines embedded in a pure ice background
    - Multi year is modelled as scattering air bubbles in a saline ice background (but brines are non-scattering in this case).
    - Lake is modelled as scattering air bubbles in a pure ice background (but brines are non-scattering in this case).

    First year and mutli year are equivalent only if scattering and porosity are nulls. It is important to understand that in multi-year ice
    scattering by brine pockets is neglected because scattering is due to air bubbles and the emmodel implemented up to now are not able to deal with
    three-phase media.

    :param ice_type: Ice type. Options are "firstyear", "multiyear", "lake"
    :param thickness: thicknesses of the layers in meter (from top to bottom). The last layer thickness can be "numpy.inf" for a semi-infinite layer.
    :param add_water_substrate: Adds a substrate made of water below the ice column.
    Possible arguments are True (default) or False. If True looks for ice_type to determine if a saline or fresh water layer is added and/or uses the
    optional arguments 'water_temperature', 'water_salinity' of the water substrate.
    :param substrate: if add_water_substrate is False, the substrate can be presribed with this argument.
    :param interface: type of interface, flat/fresnel is the default

    All the other optional arguments are passed for each layer to the function :py:func:`~smrt.inputs.make_medium.make_ice_layer`.
    The documentation of this function describes in detail the parameters used/required depending on ice_type.

"""

    # add a substrate underneath the ice (if wanted):
    if add_water_substrate:
        wp = water_parameters(ice_type, **kwargs)

        # create a permittivity_function that depends only on frequency and temperature by setting other arguments
        permittivity_model = lambda f, t: wp.water_permittivity_model(f, t, wp.water_salinity)
        substrate = Flat(wp.water_temperature, permittivity_model)
    else:
        substrate = substrate

    sp = Snowpack(substrate=substrate)


    for i, dz in enumerate(thickness):

        layer = make_ice_layer(ice_type,
                               dz, temperature=get(temperature, i),
                               salinity=get(salinity, i),
                               microstructure_model=get(microstructure_model, i),
                               brine_inclusion_shape=get(brine_inclusion_shape, i),
                               brine_volume_fraction=get(brine_volume_fraction, i),
                               porosity=get(porosity, i),
                               density=get(density, i),
                               brine_permittivity_model=get(brine_permittivity_model, i),
                               ice_permittivity_model=get(ice_permittivity_model, i),
                               saline_ice_permittivity_model=get(saline_ice_permittivity_model, i),
                               **get(kwargs, i))
        sp.append(layer, get(interface, i))

    return sp


def make_ice_layer(ice_type,
                   layer_thickness,
                   temperature,
                   salinity,
                   microstructure_model,
                   brine_inclusion_shape='spheres',
                   brine_volume_fraction=None,
                   porosity=0,
                   density=None,
                   brine_permittivity_model=None,
                   ice_permittivity_model=None,
                   saline_ice_permittivity_model=None,
                   **kwargs):
    
    """Make an ice layer for a given microstructure_model (see also :py:func:`~smrt.inputs.make_medium.make_ice_column` to create many layers).
    The microstructural parameters depend on the microstructural model and should be given as additional arguments to this function. To know which parameters are required or optional,
    refer to the documentation of the specific microstructure model used.

    :param layer_thickness: thickness of ice layer in m
    :param temperature: temperature of layer in K
    :param salinity: (firstyear and multiyear) salinity in kg/kg (see PSU constant in smrt module)
    :param brine_inclusion_shape: (firstyear and multiyear) assumption for shape of brine inclusions (so far, "spheres" and "random_needles" (i.e. elongated ellipsoidal inclusions) are implemented)
    :param brine_volume_fraction: (firstyear and multiyear) brine / liquid water fraction in sea ice, optional parameter, if not given brine volume fraction is calculated from temperature and salinity in ~.smrt.permittivity.brine_volume_fraction
    :param density: (multiyear) density of ice layer in kg m :sup:`-3`
    :param porosity: (mutliyear and fresh) air porosity of ice layer (0..1). Default is 0.
    :param ice_permittivity_model: (all) pure ice permittivity formulation (default is ice_permittivity_matzler06)
    :param brine_permittivity_model: (firstyear and multiyear) brine permittivity formulation (default is brine_permittivity_stogryn85)
    :param saline_ice_permittivity_model: (multiyear) model to mix ice and brine. The default uses polder van staten and ice_permittivity_model and brine_permittivity_model. It is highly recommanded
    to use the default.
    :param kwargs: other microstructure parameters are given as optional arguments (in Python words) but may be required (in SMRT words).
    See the documentation of the microstructure model.

    :returns: :py:class:`Layer` instance
"""

    # common setup
    if brine_volume_fraction is None:
        brine_volume_fraction = brine_volume(temperature, salinity)

    if brine_permittivity_model is None:
        brine_permittivity_model = brine_permittivity_stogryn85 # default brine permittivity model

    if density is None:
        density = bulk_ice_density(temperature, salinity, porosity)
    elif porosity == 0:
        porosity = 1. - density / bulk_ice_density(temperature, salinity, porosity=0)
    else:
        raise SMRTError("Setting density and porosity is invalid")

    # specific setup
    if ice_type == "firstyear":
        # scatterers permittivity
        eps_2 = brine_permittivity_model

        # background permittivity
        if ice_permittivity_model is None:
            # 'must import this here instead of the top of the file because of cross-dependencies' is what it says above, so I did the same...
            eps_1 = ice_permittivity_maetzler06
        else:
            eps_1 = ice_permittivity_model

        # fractional volume of brine
        frac_volume = brine_volume_fraction

        # shape of brine
        inclusion_shape = brine_inclusion_shape

        if saline_ice_permittivity_model is not None:
            raise SMRTError("Setting saline_ice_permittivity_model is invalid for firstyear seaice")

    elif ice_type == "multiyear":

        # scatterers permittivity
        eps_2 = PERMITTIVITY_OF_AIR

        if saline_ice_permittivity_model is None:
            eps_1 = ice_permittivity_maetzler06
        else:
            eps_1 = saline_ice_permittivity_pvs_mixing

        # fractional volume of air
        frac_volume = porosity

        # shape of bubbles
        inclusion_shape = 'spheres'

    elif ice_type == "fresh":
        # scatterers permittivity
        eps_2 = PERMITTIVITY_OF_AIR

        # background permittivity
        if ice_permittivity_model is None:
            # 'must import this here instead of the top of the file because of cross-dependencies' is what it says above, so I did the same...
            eps_1 = ice_permittivity_maetzler06
        else:
            eps_1 = ice_permittivity_model

        # fractional volume of air
        frac_volume = porosity

        # shape of bubbles
        inclusion_shape = 'spheres'

        if saline_ice_permittivity_model is not None or brine_permittivity_model is not None \
                or brine_volume_fraction is not None or salinity > 0:
            raise SMRTError("Setting any saline or brine parameter is invalid for fresh ice")

    else:
        raise SMRTError("Unknown ice_type. Must be firstyear, multiyear, or fresh")

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
    lay.brine_volume_fraction = brine_volume_fraction
    lay.density = density  # just for information
    lay.brine_inclusion_shape = brine_inclusion_shape
    lay.porosity = porosity  # just for information
    lay.inclusion_shape = inclusion_shape  # shape of inclusions (air or brine depending on ice_type)
    lay.ice_type = ice_type  # just for information

    return lay


def water_parameters(ice_type, **kwargs):
    """Make a semi-infinite water layer.
    :param ice_type: ice_type is used to determine if a saline or fresh water layer is added
    Optional arguments are 'water_temperature', 'water_salinity' and 'water_depth' of the water layer.
    """

    if ice_type in ['firstyear', 'multiyear']:
        water_temperature = FREEZING_POINT - 1.8
        water_salinity = 0.032  # = 0.032kg/kg = 32PSU; somewhat arbitrary value, fresher than average ocean salinity, reflecting lower salinities in polar regions
    elif ice_type == 'fresh':
        water_temperature = FREEZING_POINT
        water_salinity = 0.
    else:
        raise SMRTError("'medium' must be set to one of the following: True (default), 'ocean', 'fresh'. Additional optional arguments for function make_ice_column are 'water_temperature', 'water_salinity' and 'water_depth'.")

    water_depth = 10.  # arbitrary value of 10m thickness for the water layer, microwave absorption in water is usually high, so this represents an infinitely thick water layer

    # override the following variable if set
    WaterParameter = collections.namedtuple("WaterParameter", ('water_temperature', 'water_salinity', 'water_permittivity_model'))

    wp = WaterParameter(water_temperature=kwargs.get('water_temperature', water_temperature),
                        water_salinity=kwargs.get('water_salinity', water_salinity),
                        water_permittivity_model=seawater_permittivity_klein76)
    return wp


def bulk_ice_density(temperature, salinity, porosity):
    """
    Computes bulk density of sea ice (in kg m :sup:`-3`), when considering the influence from  brine, solid salts, and air bubbles in the ice.
    Formulation from Cox & Weeks (1983): Equations for determining the gas and brine volumes in sea ice samples, J Glac. Developed for temperatures between -2--30oC.

    For higher temperatures (>2oC) is used the formulation from Lepparanta & Manninen (1988): The brine and gas content of sea ice with attention to low salinities and high temperatures.

    :param temperature: Temperature in K
    :param salinity: salinity in ppt
    :param porosity: Fractional volume of air inclusions
    :returns: Density of ice mixture

    """

    Tc = temperature - FREEZING_POINT

    if Tc > -2.0:
        alpha = np.array([-4.1221e-2, -18.407, 5.8402e-1, 2.1454e-1])
        beta = np.array([9.0312e-2, -1.6111e-2, 1.2291e-4, 1.3603e-4])

    elif Tc <= -22.9:
        alpha = np.array([-4.732, -22.45, -6.397e-1, -1.074e-2])
        beta = np.array([8.903e-2, -1.763e-2, -5.33e-4, -8.801e-6])

    else:
        alpha = np.array([9.899e3, 1.309e3, 55.27, 7.160e-1])
        beta = np.array([8.547, 1.089, 4.518e-2, 5.819e-4])

    F1 = alpha[0] + alpha[1] * Tc + alpha[2] * Tc ** 2 + alpha[3] * Tc ** 3
    F2 = beta[0] + beta[1] * Tc + beta[2] * Tc ** 2 + beta[3] * Tc ** 3

    # Density of pure ice, C&W p. 311:
    rho_ice = 0.917 - 1.403e-4 * Tc  # in g/cm^3

    # Density of mixture:
    rho = (1. - porosity) * (rho_ice * F1 / (F1 - rho_ice * salinity * F2)) * 1e3  # in kg/m3 (eq. 15, C&W, 1983)
    return rho


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
