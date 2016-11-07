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
    def get(x, i):  # function to take the i-eme value in an array or dict of array. Can deal with scalar as well

        if isinstance(x, six.string_types):
            return x
        elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            return x.values[i]
        if isinstance(x, collections.Sequence) or isinstance(x, np.ndarray):
            return x[i]
        elif isinstance(x, dict):
            return {k: get(x[k], i) for k in x}
        else:
            return x

    sp = Snowpack(substrate=substrate)

    for i, dz in enumerate(thickness):
        layer = make_snow_layer(dz, get(microstructure_model, i),
                                density=get(density, i),
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

    return lay
