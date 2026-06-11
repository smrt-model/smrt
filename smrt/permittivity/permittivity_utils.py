from typing import Callable

from smrt.core.plugin import import_function


def permittivity_function(permittivity_model: str | Callable) -> callable:
    """Return a permittivity function based on its name.

    Args:
        permittivity_model: name of the model

    Example to get the Maetzler (2006) ice permittivity function, do:
        perm_func = permittivity_model("ice_permittivity_maetzler06")

    """
    if isinstance(permittivity_model, str):
        # get the function by model name
        try:
            modulename, _ = permittivity_model.split("_permittivity_")
        except ValueError:  # unpack problem
            raise ValueError(
                f"The permittivity model {permittivity_model} has not a valid name. It must match the pattern "
                "<modulename>_permittivity_<something>."
            )

        return import_function("permittivity", modulename, permittivity_model)
    else:
        # callable or scalar or others are returned as is
        return permittivity_model
