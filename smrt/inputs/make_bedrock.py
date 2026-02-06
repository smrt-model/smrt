# coding: utf-8

"""This module provides a function to build bedrock model and some bedrock permittivity formulae.

To create a substrate, use a helper function such as `make_bedrock_substrate`. This function is able to
automatically load a specific bedrock model.

Example::

    from smrt.inputs.make_bedrock import make_bedrock_substrate
    bedrock = make_bedrock_substrate("flat", "granite_hartlieb16", temperature=270)

"""

from smrt.core.error import SMRTError
from smrt.core.interface import Substrate, get_substrate_model
from smrt.permittivity import permittivity_function

def make_bedrock(
    *args,
    **kwargs,
) -> Substrate:
    DeprecationWarning(
        "make_bedrock is deprecated and will be removed in future versions. Please use make_bedrock_substrate instead."
    )
    return make_bedrock_substrate(*args, **kwargs)

def list_bedrock_permittivity_models():
    """
    List all available bedrock permittivity models.
    """
    from smrt.permittivity import bedrock
    return [name.removeprefix("bedrock_permittivity_")
            for name in dir(bedrock)
            if name.startswith("bedrock_permittivity_")]

def make_bedrock_substrate(
    substrate_model,
    permittivity_model,
    temperature,
    **kwargs,
) -> Substrate:
    """
    Construct a bedrock substrate instance based on a given surface electromagnetic model, a permittivity model and temperature.
    This function returns a substrate and can only be used as a bottom boundary condition in a snowpack.

    Args:
        substrate_model: Name of substrate model, can be a class or a string. e.g. flat, rough_choudhury79...
        permittivity_model: Permittivity model to use. Can be a name ("granite_hartlieb16", "frozen_bedrock_tulaczyk20"),
            a function of frequency and temperature or a complex value.
        temperature: Temperature of the bedrock in K.
        **kwargs: Geometrical parameters depending on the substrate_model. Refer to the document of each model to see the
            list of required and optional parameters. Usually, it is roughness_rms, corr_length, ...

    Returns:
        Instance of the bedrock substrate model.

    Example::

        bedrock = make_bedrock_substrate("flat", 3.15, temperature=270)
        bedrock = make_bedrock_substrate("flat", 3.15+0.7j, temperature=270)
        bedrock = make_bedrock_substrate("rough_choudhury79", "granite_hartlieb16", temperature=270, roughness_rms=0.1)
    """

    # process the permittivity_model argument
    if isinstance(permittivity_model, str):
        if "_permittivity_" in permittivity_model:
            permittivity_model = permittivity_function(permittivity_model)
        else:
            # try to prepend bedrock_permittivity_
            try:
                permittivity_model = permittivity_function("bedrock_permittivity_" + permittivity_model)
            except (ValueError, SMRTError):
                raise SMRTError(f"The bedrock permittivity model '{permittivity_model}' is not recognized")

    # process the substrate_model argument
    if not isinstance(substrate_model, type):
        substrate_model = get_substrate_model(substrate_model)

    # create the instance
    return substrate_model(temperature, permittivity_model, **kwargs)
