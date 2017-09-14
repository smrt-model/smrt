# coding: utf-8

"""This module  implements the base class for all the substrate models.
To create a substrate, it is recommended to use help functions such as :py:func:`~smrt.inputs.make_soil.make_soil` rather than the class constructor.

"""

from .error import SMRTError
from .plugin import import_class


class Substrate(object):
    """ Abstract class for substrate at the bottom of the snowpack.
It provides argument handling and calculation of the permittivity constant for soil case.
"""

    def __init__(self, temperature=None, permittivity_model=None, **kwargs):
        """ Build the substrate at the base of the snowpack

        :param temperature: temperature of the base of the snowpack. Can be the effective temperature if the substrate is slightly transparent

        :param permittivity_model: a function that return the permittivity as a function of frequency and temperature. Can also be a numerical value.

        :param **kwargs: other parameters such as roughness_rms, corr_length, Q, N, etc are required or optional depending on the model. See the document of the model.

        """

        self.temperature = temperature
        self.permittivity_model = permittivity_model  # this is a function, so it automatically becomes a method of substrate

        for arg in self.args:
            if arg in kwargs:
                setattr(self, arg, kwargs.get(arg))
            else:
                raise SMRTError("Parameter %s must be specified" % arg)

        for arg in self.optional_args:
            setattr(self, arg, kwargs.get(arg, self.optional_args[arg]))

    def permittivity(self, frequency):
        """compute the permittivity for the given frequency using permittivity_model. This method returns None when no permittivity model is
        available. This must be handled by the calling code and interpreted suitably."""

        if self.permittivity_model is None:
            return None
        else:
            return self.permittivity_model(frequency, self.temperature)


def get_substrate_model(substrate_model):
    """return the class corresponding to the substrate model called name. This function imports the correct module if possible and returns the class"""

    return import_class(substrate_model, root="smrt.substrate")

