# coding: utf-8

"""
The helper functions in this module are used to create snowpacks, sea-ice and other media. They are user-friendly and recommended
for most usages. Extension of these functions is welcome on the condition they keep a generic structure.

The function :py:func:`~smrt.inputs.make_medium.make_snowpack` is the first entry point the user should consider to build a snowpack.
For example::

    from smrt import make_snowpack

    sp = make_snowpack([1000], density=[300], microstructure_model='sticky_hard_spheres', radius=[0.3e-3], stickiness=0.2)

creates a semi-infinite snowpack made of sticky hard spheres with radius 0.3mm and stickiness 0.2.
The :py:obj:`~smrt.core.Snowpack` object is in the `sp` variable.

Note that any layer with zero thickness is completely removed in most of these functions (as well as its top interface),
and a transparent layer is added if the resulting medium does not have any layer. This allows simulation of bare soil and bare ice
more easily. It is important to understand that any layer with non-zero thickness, even much smaller than the wavelength, even
10^-20 m, has an impact in the radiative transfer framework due to the reflection, transmission and refraction. In reality,
and according to the wave theory such sub-wavelength layers and their interface should have reduced or close to zero impact.
It is the responsability of the user to ensure that such thin layers (less than a quarter of wavelength) are removed from
the snowpack. Alternatively setting the `process_coherent_layers` option when using the
`smrt.rtsolver.dort` solver allows to deal with sub-wavelength layers provided they are isolated between two thick layers.

Note that `make_snowpack` is directly imported from `smrt` instead of `smrt.inputs.make_medium`. This feature is for convenience.

"""

import itertools
import collections
import inspect
from warnings import warn

import numpy as np
import pandas as pd

from smrt.core.snowpack import Snowpack
from smrt.core.interface import make_interface
from smrt.core.plugin import import_class
from smrt.core.globalconstants import FREEZING_POINT, DENSITY_OF_ICE, DENSITY_OF_WATER, PERMITTIVITY_OF_AIR, PSU
from smrt.core.layer import get_microstructure_model, Layer
from smrt.core.error import SMRTError, smrt_warn
from smrt.core import lib
from smrt.permittivity.ice import ice_permittivity_maetzler06  # default pure ice permittivity model
from smrt.permittivity.brine import brine_volume_cox83_lepparanta88
from smrt.permittivity.saline_water import seawater_permittivity_klein76, brine_permittivity_stogryn85
from smrt.permittivity.saline_ice import saline_ice_permittivity_pvs_mixing

from smrt.substrate.flat import Flat


def make_medium(data, surface=None, interface=None, substrate=None, **kwargs):
    """build a multi-layered medium using a pandas DataFrame (or a dict that can be transformed into a DataFrame) and optinal arguments.
    The 'medium' column (or key) in data indicates the medium type: 'snow' or 'ice'. If not given, it defaults to 'snow'.
    'data' must contain enough information to build either a snowpack or an ice_column. The minimum requirements are:
    - for a snowpack: ('z' or 'thickness'), 'density', 'microstructure_model' and the arguments required by the microstructural_model.
    - for a ice column: ice_type, ('z' or 'thickness'), 'temperature', 'salinity', 'microstructure_model' and the arguments required by
    the microstructural_model.

    When reading a dataframe from disk for instance, it is convenient to use df.rename(columns={...}) to map the column names of the file
    to the column names required by SMRT.

    if 'z' is given, the thickness is deduced using :py:meth:`~smrt.core.inputs.make_medium.compute_thickness_from_z`.

    .. warning::
        Using this function is a bit dangerous as any unrecognized column names are silently ignored.
        For instance, a column named 'Temperature' is ignore (due to the uppercase), and the temperature in the snowpack
        will be set to its default value (273.15 K). This issue applies to any optional argument. Double ckeck the spelling of the columns.

    .. note::
        `make_medium` create layers using all the columns in the dataframe. It means that any column name becomes an attribute of
        the layer objects, even if not recognized/used by SMRT. This can be seen as an interesting feature to store information in layers,
        but this is also dangerous if column names collide with internal layer attributes or method names. For this reason,
        this function is unsecure if the snowpack data are pulled from the internet. Always check the content of the file, and it is recommended
        to drop all the unnecessary columns with df.drop(columns=[...])) before calling make_medium.

"""

    if isinstance(data, dict):
        # should be a dataframe, let's try to make one
        data = pd.DataFrame(data)

    if "z" in data:
        data = data.copy()
        data['thickness'] = compute_thickness_from_z(data['z'])

    if kwargs:
        data = data.copy()
        for k in kwargs:
            data[k] = kwargs[k]

    # group layers by medium type
    if 'medium' not in data:
        medium_chunks = [('snow', data)]
    else:
        medium_chunks = ((group, data.iloc[list(imedia)]) for group, imedia in
                         itertools.groupby(range(len(data)), lambda i: data.iloc[i]['medium']))

    # iterate on media chunk
    medium_list = []

    for medium, group_data in medium_chunks:

        if medium == 'snow':
            required_args = ['thickness', 'microstructure_model', 'density']
            func = make_snowpack
        elif medium == 'ice':
            required_args = ['ice_type', 'thickness', 'temperature', 'microstructure_model']
            func = make_ice_column
        else:
            raise SMRTError("Unknown medium '%s' in data" % medium)

        # compute required and optional arguments
        args = [group_data[a] for a in required_args]
        kwargs = {a: group_data[a] for a in data.columns if a not in required_args}

        m = func(*args, surface=surface, interface=interface, substrate=substrate, **kwargs)
        medium_list.append(m)

    # stack the media
    return sum(medium_list)


def make_snowpack(thickness,
                  microstructure_model,
                  density,
                  interface=None,
                  surface=None,
                  substrate=None,
                  atmosphere=None,
                  **kwargs):
    """
    Build a multi-layered snowpack. Each parameter can be an array, list or a constant value.

    :param thickness: thicknesses of the layers in meter (from top to bottom). The last layer thickness can be "numpy.inf"
        for a semi-infinite layer. Any layer with zero thickness is removed.
    :param microstructure_model: microstructure_model to use (e.g. sticky_hard_spheres or independent_sphere or exponential).
    :param surface: type of surface interface, flat/fresnel is the default.  If surface and interface are both set,
        the interface must be a constant refering to all the "internal" interfaces.
    :param interface: type of interface, flat/fresnel is the default. It is usually a string for the interfaces
        without parameters (e.g. Flat or Transparent) or is created with :py:func:`~smrt.core.interface.make_interface` in more complex cases.
        Interface can be a constant or a list. In the latter case, its length must be the same as the number of layers,
        and interface[0] refers to the surface interface.
    :param density: densities of the layers.
    :param substrate: set the substrate of the snowpack. Another way to add a substrate is to use the + operator
        (e.g. snowpack + substrate).
    :param \\**kwargs: All the other parameters (temperature, microstructure parameters, emmodel, etc.) are given as optional arguments
        (e.g. temperature=[270, 250]).
        They are passed for each layer to the function :py:func:`~smrt.inputs.make_medium.make_snow_layer`.
        Thus, the documentation of this function is the reference. It describes precisely the available parameters.
        The microstructure parameter(s) depend on the microstructure_model used and is documented in each microstructure_model module.

    e.g.::

        sp = make_snowpack([1, 10], "exponential", density=[200,300], temperature=[240, 250], corr_length=[0.2e-3, 0.3e-3])

"""

    sp = Snowpack(substrate=substrate, atmosphere=atmosphere)

    if not isinstance(thickness, collections.abc.Iterable):
        raise SMRTError("The thickness argument must be iterable, that is, a list of numbers, numpy array or pandas Series or DataFrame.")

    lib.check_argument_size(density, len(thickness), "density")
    lib.check_argument_size(kwargs, len(thickness))

    if surface is not None and lib.is_sequence(interface):
        raise SMRTError("Setting both 'surface' and 'interface' arguments is ambiguous when inteface is a list or any sequence.")

    for i, dz in enumerate(thickness):
        if dz <= 0:
            continue
        layer = make_snow_layer(dz, lib.get(microstructure_model, i, "microstructure_model"),
                                density=lib.get(density, i, "density"),
                                **lib.get(kwargs, i))

        # add the interface or surface for the first non-zero layer
        linterface = lib.get(interface, i, "interface") if surface is None else surface
        surface = None
        sp.append(layer, interface=make_interface(linterface))

    # snowpack without layer is accepted as input of this function, but SMRT prefers to have one internally.
    # we make a transparent volume
    if sp.nlayer == 0:
        sp = add_transparent_layer(sp)

    return sp


def make_snow_layer(layer_thickness,
                    microstructure_model,
                    density,
                    temperature=FREEZING_POINT,
                    ice_permittivity_model=None,
                    background_permittivity_model=PERMITTIVITY_OF_AIR,
                    volumetric_liquid_water=None,
                    liquid_water=None,
                    salinity=0,
                    medium="snow",
                    ** kwargs):
    """Make a snow layer for a given microstructure_model (see also :py:func:`~smrt.inputs.make_medium.make_snowpack`
    to create many layers). The microstructural parameters depend on the microstructural model and should be given as
    additional arguments to this function. To know which parameters are required or optional, refer to the documentation
    of the specific microstructure model used.

    :param layer_thickness: thickness of snow layer in m.
    :param microstructure_model: module name of microstructure model to be used.
    :param density: density of snow layer in kg m :sup:`-3`. Includes the ice and water phases.
    :param temperature: temperature of layer in K.
    :param ice_permittivity_model: permittivity formulation of the scatterers (default is ice_permittivity_matzler87).
    :param background_permittivity_model: permittivity formulation for the background (default is air).
    :param volumetric_liquid_water: volume of liquid water with respect to the volume of snow (default=0).
    :param liquid_water: May be depreciated in the future (use instead volumetric_liquid_water): volume of liquid water
        with respect to ice+water volume (default=0). liquid_water = water_volume / (ice_volume + water_volume).
    :param salinity: salinity in kg/kg, for using PSU as unit see PSU constant in smrt module (default = 0).
    :param medium: indicate which medium the layer is made of ("snow" is a default).
        It is used when emmodel is a dictionary mapping from medium to emmodels in :py:func:`~smrt.core.model.make_model`
    :param kwargs: other microstructure parameters are given as optional arguments (in Python words) but may be required (in SMRT words).
        See the documentation of the microstructure model.

    :returns: :py:class:`SnowLayer` instance
"""

    if ice_permittivity_model is None:
        # must import this here instead of the top of the file because of cross-dependencies
        # default ice permittivity model, use ice_permittivity_maetzler06 for dry snow and add support for wet snow
        from ..permittivity.wetice import wetice_permittivity_bohren83
        ice_permittivity_model = wetice_permittivity_bohren83

    if (salinity > 0) and 'salinity' not in inspect.signature(ice_permittivity_model).parameters:
        smrt_warn("The salinity of the layer is >0 but the permittivity formulation does not depend on salinity.  " +
                  "See the module smrt.permittivity.saline_ice module.")

    eps_1 = background_permittivity_model
    eps_2 = ice_permittivity_model

    warn_mixing_formula(background_permittivity_model, "background_permittivity_model")
    warn_mixing_formula(ice_permittivity_model, "ice_permittivity_model")

    if isinstance(microstructure_model, str):
        microstructure_model = get_microstructure_model(microstructure_model)

    # if liquid_water is not None:
    #    raise smrt_warn("The argument 'liquid_water' is going to be depreciated because its definition is uncommon"
    #                    " in the snow community. Use instead volumetric_liquid_water. Check the definition")

    lay = SnowLayer(layer_thickness,
                    medium=medium,
                    microstructure_model=microstructure_model,
                    density=float(density),
                    temperature=float(temperature),
                    permittivity_model=(eps_1, eps_2),
                    salinity=float(salinity),
                    volumetric_liquid_water=volumetric_liquid_water,
                    liquid_water=liquid_water,
                    **kwargs)

    return lay


class SnowLayer(Layer):
    """Specialized Layer class for snow. It deals with the calculation of the frac_volume and the liquid_water
     from density and volumetric_liquid_water. Alternatively it is possible to set liquid_water directly but this is
     not recommended anymore.

    :meta private:
     """

    def __init__(self, *args, density=None, volumetric_liquid_water=None, liquid_water=None, **kwargs):

        frac_volume, liquid_water = SnowLayer.compute_frac_volumes(density, volumetric_liquid_water, liquid_water)

        super().__init__(*args,
                         density=density,
                         volumetric_liquid_water=volumetric_liquid_water,
                         frac_volume=frac_volume,
                         liquid_water=liquid_water,
                         **kwargs)
        self.read_only_attributes = {'density', 'volumetric_liquid_water', 'liquid_water'}

    def update(self, density=None, volumetric_liquid_water=None, liquid_water=None, **kwargs):
        """update the density and/or volumetric_liquid_water.
        This method must be used every time density and/or volumetric_liquid_water are changed.
        Setting directly the corresponding attributes of the Layer object raises an error because
        a recalculation of the frac_volume and liquid_volume is necessary every time one of these variables
        is changed.
"""

        if density is not None:
            # avoid the readonly status
            self.__dict__['density'] = density

        if volumetric_liquid_water is not None:
            # avoid the readonly status
            self.__dict__['volumetric_liquid_water'] = volumetric_liquid_water

        self.frac_volume, self.__dict__['liquid_water'] = \
            SnowLayer.compute_frac_volumes(self.density, self.volumetric_liquid_water, liquid_water)

        super().update(**kwargs)

    @staticmethod
    def compute_frac_volumes(density, volumetric_liquid_water=None, liquid_water=None):
        """compute and return the fractional volumes:
        - frac_volume =(ice+water) / (ice+water+air)
        - liquid_water =(water) / (ice+water)
    """

        if volumetric_liquid_water is not None:
            if liquid_water is not None:
                raise SMRTError("Setting both liquid_water and volumetric_liquid_water is ambiguous")
            frac_volume = (density - (DENSITY_OF_WATER - DENSITY_OF_ICE) * volumetric_liquid_water) / DENSITY_OF_ICE
            liquid_water = volumetric_liquid_water / frac_volume
        else:
            if liquid_water is None:
                liquid_water = 0
            frac_volume = float(density) / (DENSITY_OF_ICE * (1 - liquid_water) + DENSITY_OF_WATER * liquid_water)
            # volumetric_liquid_water = liquid_water * frac_volume

        # assert frac_volume == density / (DENSITY_OF_ICE * (1 - liquid_water) + DENSITY_OF_WATER * liquid_water)

        if 1 < frac_volume < 1.01:  # consider we have a small rounding error
            frac_volume = 1

        assert 0 <= frac_volume <= 1, f"the frac_volume of ice+water in snow is {frac_volume} but must be between 0 and 1."
        " Check that volumetric_liquid_water is between 0 and 1,"
        " and that density is between 0 and DENSITY_OF_ICE + (DENSITY_OF_WATER - DENSITY_OF_ICE) * volumetric_liquid_water. "

        assert 0 <= liquid_water <= 1, f"liquid_water is {liquid_water} but must be between 0 and 1."
        " Check the volumetric_liquid_water and density arguments"

        return frac_volume, liquid_water


def make_ice_column(ice_type,
                    thickness,
                    temperature,
                    microstructure_model,
                    brine_inclusion_shape='spheres',
                    salinity=0.,
                    brine_volume_fraction=None,
                    brine_volume_model=None,
                    brine_permittivity_model=None,
                    ice_permittivity_model=None,
                    saline_ice_permittivity_model=None,
                    porosity=0,
                    density=None,
                    add_water_substrate=True,
                    surface=None,
                    interface=None,
                    substrate=None,
                    atmosphere=None,
                    **kwargs):
    """Build a multi-layered ice column. Each parameter can be an array, list or a constant value.

    ice_type variable determines the type of ice, which has a big impact on how the medium is modelled and the parameters:
    - First year ice is modelled as scattering brines embedded in a pure ice background
    - Multi year ice is modelled as scattering air bubbles in a saline ice background (but brines are non-scattering in this case).
    - Fresh ice is modelled as scattering air bubbles in a pure ice background (but brines are non-scattering in this case).

    First-year and multi-year ice is equivalent only if scattering and porosity are nulls. It is important to understand that
    in multi-year ice scattering by brine pockets is neglected because scattering is due to air bubbles and the emmodel
    implemented up to now are not able to deal with three-phase media.

    :param ice_type: Ice type. Options are "firstyear", "multiyear", "fresh"
    :param thickness: thicknesses of the layers in meter (from top to bottom). The last layer thickness can be "numpy.inf"
        for a semi-infinite layer. Any layer with zero thickness is removed.
    :param temperature: temperature of ice/water in K
    :param brine_inclusion_shape: assumption for shape of brine inclusions. So far, "spheres" or "random_needles"
        (i.e. elongated ellipsoidal inclusions), and "mix" (a mix of the two) are implemented.
    :param salinity: salinity of ice/water in kg/kg (see PSU constant in smrt module). Default is 0. If neither salinity
        nor brine_volume_fraction are given, the ice column is considered to consist of fresh water ice.
    :param brine_volume_fraction: brine / liquid water fraction in sea ice. Can be a value or a function depending on temperature and salinity.
        See the module :py:mod:`smrt.permittivity.brine` for available options.
        This parameter is optional, if not given brine volume fraction is calculated from temperature and salinity in
        :py:func:`~.smrt.permittivity.brine.brine_volume_cox83_lepparanta88`.
    :param density: density of ice layer in kg m :sup:`-3`
    :param porosity: porosity of ice layer (0 - 1). Default is 0.
    :param add_water_substrate: Adds a substrate made of water below the ice column.
        Possible arguments are True (default) or False. If True looks for ice_type to determine if a saline or fresh water layer is
        added and/or uses the optional arguments 'water_temperature', 'water_salinity' of the water substrate.
    :param surface: type of surface interface, flat/fresnel is the default.  If surface and interface are both set, the interface must be
        a constant refering to all the "internal" interfaces.
    :param interface: type of interface, flat/fresnel is the default. It is usually a string for the interfaces without parameters
        (e.g. Flat or Transparent) or is created with :py:func:`~smrt.core.interface.make_interface` in more complex cases.
        Interface can be a constant or a list. In the latter case, its length must be the same as the number of layers,
        and interface[0] refers to the surface interface.
    :param substrate: if add_water_substrate is False, the substrate can be prescribed with this argument.

    All the other optional arguments are passed for each layer to the function :py:func:`~smrt.inputs.make_medium.make_ice_layer`.
    The documentation of this function describes in detail the parameters used/required depending on ice_type.

"""

    # add a substrate underneath the ice (if wanted):
    if add_water_substrate:
        wp = water_parameters(ice_type, **kwargs)

        # create a permittivity_function that depends only on frequency and temperature by setting other arguments
        def permittivity_model(f, t):
            return wp.water_permittivity_model(f, t, wp.water_salinity)
        substrate = Flat(temperature=wp.water_temperature, permittivity_model=permittivity_model)
    else:
        substrate = substrate

    sp = Snowpack(substrate=substrate, atmosphere=atmosphere)

    n = len(thickness)
    for name in ["temperature", "salinity", "microstructure_model", "brine_inclusion_shape", "brine_volume_fraction",
                 "porosity", "density", "brine_permittivity_model", "ice_permittivity_model", "saline_ice_permittivity_model",
                 "interface", "kwargs"]:
        lib.check_argument_size(locals()[name], n)

    if surface is not None and lib.is_sequence(interface):
        raise SMRTError("Setting both 'surface' and 'interface' arguments is ambiguous when inteface is a list or any sequence.")

    for i, dz in enumerate(thickness):
        if dz <= 0:
            continue
        layer = make_ice_layer(ice_type,
                               dz, temperature=lib.get(temperature, i),
                               salinity=lib.get(salinity, i),
                               microstructure_model=lib.get(microstructure_model, i),
                               brine_inclusion_shape=lib.get(brine_inclusion_shape, i),
                               brine_volume_fraction=lib.get(brine_volume_fraction, i),
                               porosity=lib.get(porosity, i),
                               density=lib.get(density, i),
                               brine_permittivity_model=lib.get(brine_permittivity_model, i),
                               ice_permittivity_model=lib.get(ice_permittivity_model, i),
                               saline_ice_permittivity_model=lib.get(saline_ice_permittivity_model, i),
                               **lib.get(kwargs, i))

        # add the interface or surface for the first non-zero layer
        linterface = lib.get(interface, i, "interface") if surface is None else surface
        surface = None
        sp.append(layer, interface=make_interface(linterface))

    # snowpack without layer is accepted as input of this function, but SMRT prefers to have one internally.
    # we make a transparent volume
    if sp.nlayer == 0:
        sp = add_transparent_layer(sp)

    return sp


def make_ice_layer(ice_type,
                   layer_thickness,
                   temperature,
                   salinity,
                   microstructure_model,
                   brine_inclusion_shape='spheres',
                   brine_volume_fraction=None,
                   brine_permittivity_model=None,
                   porosity=0,
                   density=None,
                   ice_permittivity_model=None,
                   saline_ice_permittivity_model=None,
                   medium="ice",
                   **kwargs):
    """Make an ice layer for a given microstructure_model (see also :py:func:`~smrt.inputs.make_medium.make_ice_column`
    to create many layers). The microstructural parameters depend on the microstructural model and should be given as
    additional arguments to this function. To know which parameters are required or optional, refer to the documentation
    of the specific microstructure model used.

    :param ice_type: Assumed ice type
    :param layer_thickness: thickness of ice layer in m.
    :param temperature: temperature of layer in K
    :param salinity: (firstyear and multiyear) salinity in kg/kg (see PSU constant in smrt module)
    :param brine_inclusion_shape: (firstyear and multiyear) assumption for shape of brine inclusions (so far,
        "spheres" and "random_needles" (i.e. elongated ellipsoidal inclusions), and "mix_spheres_needles" are implemented)
    :param brine_volume_fraction: brine / liquid water fraction in sea ice. Can be a value or a function depending on temperature and salinity.
        See the module :py:mod:`smrt.permittivity.brine` for available options.
        This parameter is optional, if not given brine volume fraction is calculated from temperature and salinity in
        :py:func:`~.smrt.permittivity.brine.brine_volume_cox83_lepparanta88`.
    :param brine_permittivity_model: (firstyear and multiyear) brine permittivity formulation
        (default is brine_permittivity_stogryn85)
    :param density: (multiyear) density of ice layer in kg m :sup:`-3`. If not given, density is calculated from temperature,
        salinity and ice porosity.
    :param porosity: (mutliyear and fresh) air porosity of ice layer (0..1). Default is 0.
    :param ice_permittivity_model: (all) pure ice permittivity formulation
        (default is ice_permittivity_matzler06 for firstyear and fresh, and saline_ice_permittivity_pvs_mixing for multiyear)
    :param saline_ice_permittivity_model: (multiyear) model to mix ice and brine. The default uses polder van staten and
        ice_permittivity_model and brine_permittivity_model. It is highly recommanded to use the default.
    :param kwargs: other microstructure parameters are given as optional arguments (in Python words) but may be required (in SMRT words).
    :param medium: indicate which medium the layer is made of ("ice" is a default).
        It is used when emmodel is a dictionary mapping from medium to emmodels in :py:func:`~smrt.core.model.make_model`

    See the documentation of the microstructure model.

    :returns: :py:class:`Layer` instance
"""

    # common setup
    if ice_type in ['firstyear', 'multiyear']:
        if brine_volume_fraction is None:
            brine_volume_fraction = brine_volume_cox83_lepparanta88(temperature, salinity)
        if callable(brine_volume_fraction):
            # call it and get a value
            brine_volume_fraction = brine_volume_fraction(temperature, salinity)

        if brine_permittivity_model is None:
            brine_permittivity_model = brine_permittivity_stogryn85  # default brine permittivity model

        warn_mixing_formula(brine_permittivity_model, "brine_permittivity_model")
        warn_mixing_formula(saline_ice_permittivity_model, "saline_ice_permittivity_model")

    warn_mixing_formula(ice_permittivity_model, "ice_permittivity_model")

    if density is None:
        density = bulk_ice_density(temperature, salinity, porosity)
    elif porosity == 0:
        porosity = np.clip(1. - density / bulk_ice_density(temperature, salinity, porosity=0), 0., 1.)
    else:
        raise SMRTError("Setting density and porosity is invalid")

    # specific setup
    if ice_type == "firstyear":
        # scatterers permittivity
        eps_2 = brine_permittivity_model

        # background permittivity
        if ice_permittivity_model is None:
            # 'must import this here instead of the top of the file because of cross-dependencies' is what it says above,
            # so I did the same...
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

        # background permittivity
        if saline_ice_permittivity_model is None:
            eps_1 = saline_ice_permittivity_pvs_mixing
        else:
            eps_1 = saline_ice_permittivity_model

        # fractional volume of air
        frac_volume = porosity

        # shape of air bubbles
        inclusion_shape = 'spheres'

    elif ice_type == "fresh":
        # scatterers permittivity
        eps_2 = PERMITTIVITY_OF_AIR

        # background permittivity
        if ice_permittivity_model is None:
            # 'must import this here instead of the top of the file because of cross-dependencies' is what it says above,
            # so I did the same...
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

    if isinstance(microstructure_model, str):
        microstructure_model = get_microstructure_model(microstructure_model)

    lay = Layer(float(layer_thickness),
                medium="ice",
                microstructure_model=microstructure_model,
                frac_volume=float(frac_volume),
                temperature=float(temperature),
                permittivity_model=(eps_1, eps_2),
                inclusion_shape=inclusion_shape,
                salinity=float(salinity),
                **kwargs)

    if brine_volume_fraction is not None:
        lay.brine_volume_fraction = float(brine_volume_fraction)
        lay.brine_inclusion_shape = brine_inclusion_shape
    lay.density = float(density)  # just for information, read-only
    lay.porosity = float(porosity)  # just for information, read-only
    lay.inclusion_shape = inclusion_shape  # shape of inclusions (air or brine depending on ice_type)
    lay.ice_type = ice_type  # just for information, read-only

    lay.read_only_attributes = {'ice_type', 'density', 'porosity'}

    return lay


def make_water_body(layer_thickness=1000,
                    temperature=FREEZING_POINT,
                    salinity=0,
                    water_permittivity_model=None,
                    foam_frac_volume=0,
                    surface=None,
                    atmosphere=None,
                    substrate=None):
    """Make a water body with a single layer of water at given temperature and salinity.

    Note that water is a very strong absorber even fresh water, it is unlikely that the layers under a water body
    could be seen by microwaves. If really needed anyway, a multi-layer water body or
    a water layer on another medium (e.g. ice) can be build using the addition operator.

    Note that water has a strong real permittivity and when used in
    combinaison with the DORT solver, it is recommended to increase the `n_max_stream` option of the solver to get
    enough streams in the air (see about stream Picard et al. 2018).

    :param layer_thickness: thickness of ice layer in m. If the thickness is zero, a transparent layer is added.
    :param temperature: temperature of layer in K
    :param salinity: salinity in kg/kg (see PSU constant in smrt module)
    :param water_permittivity_model: water permittivity formulation (default is seawater_permittivity_klein76)
    :param foam_frac_volume: fractional volume of air bubbles in the water. See for instance Hwang et al. 2019.
        https://doi.org/10.1175/JPO-D-19-0061.1 . Note that the permittivity mixing formula suggested in that paper is
        different from the Polder van Santen used in most emmodels in SMRT.
    :param foam_bubble_radius: effective radius of the foam bubbles. See for instance Golbraikh and Shtemler, 2018
      doi:10.1007/s10236-018-1166-4
    :param surface: type of surface interface. Flat surface (Fresnel coefficient) is the default.
    :param substrate: the substrate under the water layer.

"""

    sp = Snowpack(substrate=substrate, atmosphere=atmosphere)  # Snowpack is not a typo, yes SMRT use Snowpacks even for water or ice.

    layer = make_water_layer(layer_thickness,
                             temperature=temperature,
                             salinity=salinity,
                             water_permittivity_model=water_permittivity_model,
                             foam_frac_volume=foam_frac_volume)
    # add the layer and the interface interface
    sp.append(layer, interface=make_interface(surface))

    if layer_thickness <= 0:
        # snowpack without layer is accepted as input of this function, but SMRT prefers to have one internally.
        # we make a transparent volume
        sp = add_transparent_layer(sp)

    return sp


def make_water_layer(layer_thickness,
                     temperature=FREEZING_POINT,
                     salinity=0,
                     water_permittivity_model=None,
                     foam_frac_volume=0,
                     foam_bubble_radius=0.1e-3,
                     **kwargs):
    """Make a water layer at given temperature and salinity.

    :param layer_thickness: thickness of ice layer in m
    :param temperature: temperature of layer in K
    :param salinity: salinity in kg/kg (see PSU constant in smrt module)
    :param water_permittivity_model: water permittivity formulation (default is seawater_permittivity_klein76)
    :param foam_frac_volume: fractional volume of air bubbles in the water. See for instance Hwang et al. 2019.
        https://doi.org/10.1175/JPO-D-19-0061.1 . Note that the permittivity mixing formula suggested in that paper is
        different from the Polder van Santen used in most emmodels in SMRT.
    :param foam_bubble_radius: effective radius of the foam bubbles. See for instance Golbraikh and Shtemler, 2018
      doi:10.1007/s10236-018-1166-4
"""
    if water_permittivity_model is None:
        water_permittivity_model = seawater_permittivity_klein76

    if foam_frac_volume == 0:
        microstructure_model = "homogeneous"
    else:
        microstructure_model = "sticky_hard_spheres"
        kwargs['radius'] = foam_bubble_radius

    lay = Layer(float(layer_thickness),
                medium="water",
                microstructure_model=get_microstructure_model(microstructure_model),
                frac_volume=foam_frac_volume,
                temperature=float(temperature),
                permittivity_model=(water_permittivity_model, 1.),
                salinity=float(salinity),
                **kwargs)

    return lay


def water_parameters(ice_type, **kwargs):
    """Make a semi-infinite water layer.

    :param ice_type: ice_type is used to determine if a saline or fresh water layer is added
        The optional arguments are 'water_temperature', 'water_salinity' and 'water_depth' of the water layer.
    """

    # prepare default
    if ice_type in ['firstyear', 'multiyear']:
        water_temperature = FREEZING_POINT - 1.8
        water_salinity = 0.032  # = 0.032kg/kg = 32PSU; somewhat arbitrary value,
        # fresher than average ocean salinity, reflecting lower salinities in polar regions
    elif ice_type == 'fresh':
        water_temperature = FREEZING_POINT
        water_salinity = 0.
    else:
        raise SMRTError("'medium' must be set to one of the following: True (default), 'ocean', 'fresh'. Additional optional arguments"
                        " for function make_ice_column are 'water_temperature', 'water_salinity' and 'water_depth'.")

    # override the following variable if set
    WaterParameter = collections.namedtuple("WaterParameter", ('water_temperature', 'water_salinity', 'water_permittivity_model'))

    wp = WaterParameter(water_temperature=kwargs.get('water_temperature', water_temperature),
                        water_salinity=kwargs.get('water_salinity', water_salinity),
                        water_permittivity_model=seawater_permittivity_klein76)
    return wp


def bulk_ice_density(temperature, salinity, porosity):
    """
    Computes bulk density of sea ice (in kg m :sup:`-3`), when considering the influence from  brine, solid salts, and
    air bubbles in the ice. Formulation from Cox & Weeks (1983): Equations for determining the gas and brine volumes in sea ice samples,
    J Glac. Developed for temperatures between -2--30째C. For higher temperatures (>2째C) is used the formulation from
    Lepparanta & Manninen (1988): The brine and gas content of sea ice with attention to low salinities and high temperatures.

    :param temperature: Temperature in K
    :param salinity: salinity in kg/kg (see PSU constant in smrt module)
    :param porosity: Fractional volume of air inclusions (0..1)
    :returns: Density of ice mixture in kg m :sup:`-3`
    """

    Tc = temperature - FREEZING_POINT

    if Tc > -2.0:
        alpha = np.array([-4.1221e-2, -18.407, 5.8402e-1, 2.1454e-1])
        beta = np.array([9.0312e-2, -1.6111e-2, 1.2291e-4, 1.3603e-4])

    elif Tc >= -22.9:
        alpha = np.array([-4.732, -22.45, -6.397e-1, -1.074e-2])
        beta = np.array([8.903e-2, -1.763e-2, -5.33e-4, -8.801e-6])

    else:
        alpha = np.array([9.899e3, 1.309e3, 55.27, 7.160e-1])
        beta = np.array([8.547, 1.089, 4.518e-2, 5.819e-4])

    F1 = np.polyval(alpha[::-1], Tc)
    F2 = np.polyval(beta[::-1], Tc)

    # Density of pure ice, C&W p. 311:
    rho_ice = 0.917 - 1.403e-4 * Tc  # in g/cm^3

    # Density of mixture:
    rho = (1. - porosity) * (rho_ice * F1 / (F1 - rho_ice * salinity * PSU**-1 * F2)) * 1e3  # in kg/m3 (eq. 15, C&W, 1983)

    if rho < 0:
        raise SMRTError("Ice density may not be negative.")

    return rho


def make_generic_stack(thickness, temperature=FREEZING_POINT, ks=0, ka=0, effective_permittivity=1,
                       interface=None,
                       substrate=None,
                       atmosphere=None):
    """
    build a multi-layered medium with prescribed scattering and absorption coefficients and effective permittivity.
    Must be used with presribed_kskaeps emmodel.

    :param thickness: thicknesses of the layers in meter (from top to bottom). The last layer thickness can be "numpy.inf" for
        a semi-infinite layer. Any layer with zero thickness is removed.
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
# """

    sp = Snowpack(substrate=substrate, atmosphere=atmosphere)

    if not isinstance(thickness, collections.abc.Iterable):
        raise SMRTError("The thickness argument must be iterable, that is, a list of numbers, numpy array or pandas Series or DataFrame.")

    for i, dz in enumerate(thickness):
        if dz <= 0:
            continue
        layer = make_generic_layer(dz,
                                   ks=lib.get(ks, i, "ks"),
                                   ka=lib.get(ka, i, "ka"),
                                   effective_permittivity=lib.get(effective_permittivity, i, "effective_permittivity"),
                                   temperature=lib.get(temperature, i, "temperature")
                                   )

        sp.append(layer, lib.get(interface, i))

    # snowpack without layer is accepted as input of this function, but SMRT prefers to have one internally.
    # we make a transparent volume
    if sp.nlayer == 0:
        sp = add_transparent_layer(sp)

    return sp


def make_generic_layer(layer_thickness, ks=0, ka=0, effective_permittivity=1, temperature=FREEZING_POINT):
    """Make a generic layer with prescribed scattering and absorption coefficients and effective permittivity.
    Must be used with presribed_kskaeps emmodel.

    :param layer_thickness: thickness of ice layer in m
    :param temperature: temperature of layer in K
    :param ks: scattering coefficient of layers in m^-1
    :param ka: absorption coefficient of layers in m^-1

    :returns: :py:class:`Layer` instance
"""

    lay = Layer(layer_thickness, temperature=temperature)

    lay.temperature = float(temperature)
    lay.effective_permittivity = effective_permittivity
    lay.ks = float(ks)
    lay.ka = float(ka)

    return lay


def add_transparent_layer(snowpack):
    """
    add a transparent layer to the snowpack

    :param snowpack: the substrate under the transparent layer.

   e.g.::

       sp = add_transparent_layer(sp)

"""

    layer = Layer(thickness=0,
                  microstructure_model=get_microstructure_model("homogeneous"),
                  frac_volume=0,
                  temperature=0,
                  permittivity_model=(1, 1))

    snowpack.append(layer, interface=make_interface("transparent"))

    return snowpack


def make_transparent_volume(substrate=None,
                            atmosphere=None):
    """
    build a transparent single-layer snowpack. This is useful to run SMRT without any 'real' layer.

    :param substrate: the substrate under the transparent layer.

   e.g.::

       sp = make_transparent_volume()

"""

    return add_transparent_layer(Snowpack(substrate=substrate, atmosphere=atmosphere))


def make_atmosphere(atmosphere_model, **kwargs):
    """Make a atmospheric single-layer using the prescribed atmosphere model.
    Warning: this function is subject to change in the future when refactoring how SMRT deals with atmosphere.

    :param atmosphere_model: the name of the model to use. The available models are in smrt.atmosphere.
    :param \\**kwargs: all the parameters used by the atmosphere_model.

"""

    atmosphere_class = import_class("atmosphere", atmosphere_model)

    return atmosphere_class(**kwargs)


def compute_thickness_from_z(z):
    """Compute the thickness of layers given the elevation z. Whatever the sign of z, the order *MUST* be from the topmost layer to the
    lowermost.

    Several situation are accepted and interpretated as follows:
    - z is positive and decreasing. The first value is the height of the surface about the ground (z=0) and z represents the top elevation
    of each layer. This is typical of the seasonal snowpack.
    - z is negative and decreasing. The first value is the elevation of the bottom of the first layer with respect to the surface (z=0).
    This is typical of a snowpack on ice-sheet.
    - z is positive and increasing. The first value is the depth of the bottom of the first layer with respect to the surface.
    This is typical of a snowpack on ice-sheet.
    - other case, when z is not monoton or is increasing with negative value raises an error.

    Because z indicate the top or the bottom of a layer depending whether z=0 is the ground or the surface,
    the value 0 can never be in z. This raises an error.

"""
    order = (np.diff(z) < 0)
    if np.any(z == 0):
        raise SMRTError("z must not include 0")
    positive = z >= 0

    if np.all(order):
        # descending z
        if np.all(positive):
            # z >0, this is typically a seasonal snowpack. z= height above ground
            z = -np.append(z.values, 0)
        else:
            # z < 0, this is typically a permanent, deep snowpack, without ground reference. z is the depth from the surface.
            z = -np.insert(z.values, 0, 0)

    elif np.any(order):
        # ascending z
        if np.all(positive):
            # ascending z and z > 0, this is typically a permanent, deep snowpack, without ground reference.
            # z is the depth from the surface.
            z = np.insert(z.values, 0, 0)
        else:
            # this is unusual
            raise SMRTError("z is ascending and has negative values, which an ambiguous situation")

    else:
        raise SMRTError("The z argument is not sorted")

    return np.diff(z)


def warn_mixing_formula(permittivity_model, name):

    if not callable(permittivity_model):
        return

    signature = inspect.signature(permittivity_model).parameters
    if ('density' in signature) or ('frac_volume' in signature):
        smrt_warn(f"""The permittivity model set for the {name} argument seems to be a mixing formula. Such formula should
        not be used in this function but rather using derived_IBA or derive_SymSCE or equivalent functions. Check the
        module documentation of the permittivity model.""",
             stacklevel=2)
