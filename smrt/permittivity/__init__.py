"""
This package provides many formulations for the permititivity of various materials (ice, water, ...) or for mixing
formulae. It is worth noting that the formers are to be used as input of the functions in py:mod:`~smrt.inputs` in order
to prescribe the scatterers and background permittivity, while the latters are to be used in py:mod:`~smrt.emmodels` to
reformulate how the effective permittivity is calculated. This latter usage is very specific and should not concern most
users. See :py:func:`smrt.emmodel.symsce_torquato21.derived_SymSCETK21` and
:py:func:`smrt.emmodel.iba.derived_IBA`.


.. admonition::  **For developers**

    To add a new permittivity function proceed as follows:

    1. To add a new permittivity formulation add a function either in an existing file or
    in a new file (recommended for testing). E.g. for salty ice permittivity formulations should be in saltyice.py and so on.

    2. Any function defining a permittivity model must declare the mapping
    between the layer properties and the arguments of the function (see ice.py for examples).
    It means that the arguments of the function must be listed (in order) in the @required_layer_properties
    decorator. In most cases, the name of the arguments should be the same as a properties, but
    this is not strictly necessary, only the order matters. E.g.::

            @required_layer_properties("temperature", "salinity")
            def permittivity_something(frequency, t, s):

    maps the layer property "temperature" to the argument "t" of the function (and "salinity" to s)
    However, we recommend to change t into temperature for sake of clarity.

    For curious ones, this declaration is required because the function can be called either with its arguments (normal case)
    or with only two arguments like this (frequency, layer). In this latter case, the arguments required by the original function
    are automatically extracted from the layer attributes (=properties) based on the declaration in @required_layer_properties.
    This complication is necessary because there is no way in Python to inspect the name of the arguments of
    a function, so the need for explicit declaration.

    3. to use the new function, import the module (e.g. from smrt.permittivity.ice import permittivity_something) and
    pass this function to :py:func:`smrt.core.snowpack.make_snowpack` or :py:func:`smrt.core.layer:make_snow_layer`.

"""

