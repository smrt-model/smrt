# coding: utf-8

"""This module provides a function to build soil model and some soil permittivity formulae.

To create a substrate, use or implement a helper function such as `make_soil`. This function is able to
automatically load a specific soil model and provides some soil permittivity formulae as well.

Example::

    from smrt import make_soil
    soil = make_soil("soil_wegmuller", "dobson85", moisture=0.2, sand=0.4, clay=0.3, dry_matter=1100, roughness_rms=1e-2)

It is recommended to first read the documentation of `make_soil` and then explore the different types of soil
models.
"""

from functools import partial

from smrt.core import lib

# local import
from smrt.core.error import SMRTError, smrt_warn
from smrt.core.interface import Substrate, get_substrate_model, make_interface
from smrt.core.layer import Layer, get_microstructure_model
from smrt.core.snowpack import Snowpack
from smrt.inputs.make_medium import add_transparent_layer
from smrt.permittivity import permittivity_function
from smrt.permittivity.soil import (
    soil_permittivity_dobson85,
    soil_permittivity_dobson85_peplinski95,
    soil_permittivity_hut,
    soil_permittivity_montpetit08,
)


def make_soil(
    *args,
    **kwargs,
) -> Substrate:
    DeprecationWarning(
        "make_soil is deprecated and will be removed in future versions. Please use make_soil_substrate instead."
    )
    return make_soil_substrate(*args, **kwargs)


def make_soil_substrate(
    substrate_model,
    permittivity_model,
    temperature,
    moisture=None,
    sand=None,
    clay=None,
    dry_matter=None,
    **kwargs,
) -> Substrate:
    """
    Construct a soil substrate instance based on a given surface electromagnetic model, a permittivity model and parameters.

    This function returns a substrate and can only be used as a bottom boundary condition in a snowpack or soil-snowpack model.
    See :py:func:`~smrt.inputs.make_soil.make_soil_layer` and :py:func:`~smrt.inputs.make_soil.make_soil_column` functions
    if you want a soil layer or several soil layers.

    Args:
        substrate_model: Name of substrate model, can be a class or a string. e.g. fresnel, wegmuller...
        permittivity_model: Permittivity model to use. Can be a name ("hut_epss", "dobson85_peplinski95", "montpetit2008"),
            a function of frequency and temperature or a complex value.
        temperature: Temperature of the soil.
        moisture: Soil moisture in m^3 m^-3 to compute the permittivity. This parameter is used depending on the permittivity_model.
        sand: Soil relative sand content. This parameter is used or not depending on the permittivity_model.
        clay: Soil relative clay content. This parameter is used or not depending on the permittivity_model.
        dry_matter: Soil content in dry matter in kg m^-3. This parameter is used or not depending on the permittivity_model.
        **kwargs: Geometrical parameters depending on the substrate_model. Refer to the document of each model to see the
            list of required and optional parameters. Usually, it is roughness_rms, corr_length, ...

    Returns:
        Instance of the soil substrate model.

    Example (TOTEST)::

        bottom = substrate.make('Flat', permittivity_model=complex('6-0.5j'))
        bottom = substrate.make('Wegmuller', permittivity_model='soil', roughness_rms=0.25, moisture=0.25)
    """

    # process the permittivity_model argument
    if isinstance(permittivity_model, str):
        match permittivity_model:
            case "hut_epss":
                # return soil_permittivity_hut after setting the parameters
                if moisture is None or sand is None or clay is None or dry_matter is None:
                    raise SMRTError("The parameters moisture, sand, clay and dry_matter must be set")

                permittivity_model = partial(
                    soil_permittivity_hut,
                    moisture=moisture,
                    sand=sand,
                    clay=clay,
                    dry_matter=dry_matter,
                )
            case "dobson85":
                raise SMRTError(
                    "The model labelled 'dobson85' in SMRT was using dobson85 modified peplinski95. "
                    "To avoid this misleading name, the new recommended name is 'dobson85_peplinski95'. "
                    "In addition, the original dobson85 model is now available under the name 'dobson85_original'."
                )

            case "dobson85_original":
                # return soil_permittivity_dobson after setting the parameters
                if moisture is None or sand is None or clay is None:
                    raise SMRTError("The parameters moisture, sand, clay must be set")
                permittivity_model = partial(soil_permittivity_dobson85, moisture=moisture, sand=sand, clay=clay)

            case "dobson85_peplinski95":
                # return soil_permittivity_dobson after setting the parameters
                if moisture is None or sand is None or clay is None:
                    raise SMRTError("The parameters moisture, sand, clay must be set")
                permittivity_model = partial(
                    soil_permittivity_dobson85_peplinski95, moisture=moisture, sand=sand, clay=clay
                )

            case "montpetit2008":
                permittivity_model = soil_permittivity_montpetit08
            case _:
                if "_permittivity_" in permittivity_model:
                    permittivity_model = permittivity_function(permittivity_model)
                else:
                    raise SMRTError(f"The permittivity model '{permittivity_model}' is not recongized")
    else:
        # check that other parameters are defined
        if moisture is not None or sand is not None or clay is not None or dry_matter is not None:
            smrt_warn(
                "Setting moisture, clay, sand or dry_matter when permittivity_model is a number or function is useless"
            )

    # process the substrate_model argument
    if not isinstance(substrate_model, type):
        substrate_model = get_substrate_model(substrate_model)

    # create the instance
    return substrate_model(temperature, permittivity_model, **kwargs)


def make_soil_column(
    thickness,
    temperature,
    soil_permittivity_model,
    moisture=None,
    sand=None,
    clay=None,
    dry_matter=None,
    surface=None,
    interface=None,
    substrate=None,
    atmosphere=None,
    **kwargs,
) -> Snowpack:
    """
    Build a multi-layered soil column. Each parameter can be an array, list or a constant value.

    :param thickness: thicknesses of the layers in meter (from top to bottom). The last layer thickness can be "numpy.inf"
        for a semi-infinite layer. Any layer with zero thickness is removed.
    :param temperature: temperature of soil in K.
    :param soil_permittivity_model: Permittivity model to use. Can be a name, a function of
            frequency and temperature or a complex value.
    :param moisture: Soil moisture in m^3 m^-3 to compute the permittivity. This parameter is used depending on the permittivity_model.
    :param sand: Soil relative sand content. This parameter is used or not depending on the permittivity_model.
    :param clay: Soil relative clay content. This parameter is used or not depending on the permittivity_model.
    :param dry_matter: Soil content in dry matter in kg m^-3. This parameter is used or not depending on the permittivity_model.
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

    sp = Snowpack(
        substrate=substrate, atmosphere=atmosphere
    )  # ??????????????????????????????????????????????????????????????????????????

    n = len(thickness)
    for name in [
        "temperature",
        "moisture",
        "sand",
        "clay",
        "dry_matter",
        "interface",
        "kwargs",
    ]:
        lib.check_argument_size(locals()[name], n)

    if surface is not None and lib.is_sequence(interface):
        raise SMRTError(
            "Setting both 'surface' and 'interface' arguments is ambiguous when inteface is a list or any sequence."
        )

    for i, dz in enumerate(thickness):
        if dz <= 0:
            continue
        layer = make_soil_layer(
            soil_permittivity_model,
            dz,
            temperature=lib.get(temperature, i),
            moisture=lib.get(moisture, i),
            sand=lib.get(sand, i),
            clay=lib.get(clay, i),
            dry_matter=lib.get(dry_matter, i),
            **lib.get(kwargs, i),
        )

        # add the interface or surface for the first non-zero layer
        linterface = lib.get(interface, i, "interface") if surface is None else surface
        surface = None
        sp.append(layer, interface=make_interface(linterface))

    # snowpack without layer is accepted as input of this function, but SMRT prefers to have one internally.
    # we make a transparent volume
    if sp.nlayer == 0:
        sp = add_transparent_layer(sp)

    return sp


def make_soil_layer(
    soil_permittivity_model,
    layer_thickness,
    temperature,
    sand=None,
    clay=None,
    dry_matter=None,
    **kwargs,
) -> Layer:
    """
    Make a soil layer with given geophysical parameters

    Args:
        soil_permittivity_model: Permittivity model to use (see :py:mod:`~smrt.permittivity.soil`).  If None, the default is
            :py:func:`~smrt.permittivity.soil.soil_permittivity_dobson85_peplinski95`.
        layer_thickness: Thickness of ice layer in m.
        temperature: Temperature of layer in K.
        moisture: Soil moisture in m^3 m^-3 to compute the permittivity. This parameter is used depending on the permittivity_model.
        sand: Soil relative sand content. This parameter is used or not depending on the permittivity_model.
        clay: Soil relative clay content. This parameter is used or not depending on the permittivity_model.
        dry_matter: Soil content in dry matter in kg m^-3. This parameter is used or not depending on the permittivity_model.

    Returns:
        Layer: Instance of Layer.
    """

    # background permittivity (default = soil_permittivity_dobson85_peplinski95)
    eps_1 = permittivity_function(soil_permittivity_model) or soil_permittivity_dobson85_peplinski95

    lay = Layer(
        float(layer_thickness),
        # SMRT needs a microstructure. Here you can use the homogeneous microstructure if you don't want scattering.
        microstructure_model=get_microstructure_model(
            "homogeneous"
        ),  # SMRT needs a microstructure. Here you can use the homogeneous microstructure if you don't want scattering.
        temperature=float(temperature),
        frac_volume=0,  # Almost all microstructure need a frac_volume. frac_volume=0 means that the layer is empty of scatteres, it only contains the background
        permittivity_model=(
            eps_1,
            1,
        ),
        sand=sand,
        clay=clay,
        dry_matter=dry_matter,
        **kwargs,
    )

    # lay.read_only_attributes = {"ice_type", "density", "porosity"}   # <---- GHI: remove this

    return lay
