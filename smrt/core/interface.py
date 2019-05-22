# coding: utf-8

"""This module  implements the base class for all the substrate models.
To create a substrate, it is recommended to use help functions such as :py:func:`~smrt.inputs.make_soil.make_soil` rather than the class constructor.

"""

import inspect
from functools import wraps

from .error import SMRTError
from .plugin import import_class
from ..interface.flat import Flat  # core should not depend on something defined in interface...
from smrt.core import lib


def make_interface(inst_class_or_modulename, broadcast=True, **kwargs):
    """return an instannce class corresponding to the interface model.

    This function import the correct module if necessary and if possible and return the class. It is used internally and should not be needed for normal usage.

    :param class_or_modulename: a class or name of the python module in smrt/interface
    """

    # import the module
    if inst_class_or_modulename is None:
        interface_cls = Flat
    elif isinstance(inst_class_or_modulename, str):
        interface_cls = import_class("interface", inst_class_or_modulename)
    elif inspect.isclass(inst_class_or_modulename):
        interface_cls = inst_class_or_modulename
    elif hasattr(inst_class_or_modulename, "specular_reflection_matrix"):
        # we have an instance... good we can directly return it
        return inst_class_or_modulename
    else:
        raise SMRTError("The interface must be either the name of a module in the smrt.interface directory, or a class that implements the interface behavior.")

    if broadcast and kwargs:
        l = [len(k) for k in kwargs.values() if lib.is_sequence(k)]
        if l:
            return [interface_cls(**lib.get(kwargs, i)) for i in range(max(l))]

    return interface_cls(**kwargs)  # try to create it without argument



class Interface(object):
    """ Abstract class for interface between layer or at the bottom of the snowpack.
It provides argument handling.
"""
    args = []
    optional_args = {}

    def __init__(self, **kwargs):
        """ Build the interface

        :param **kwargs: parameters such as roughness_rms, corr_length, Q, N, etc are required or optional depending on the model. See the document of the model.

        """

        for arg in self.args:
            if arg in kwargs:
                setattr(self, arg, kwargs.get(arg))
            else:
                raise SMRTError("Parameter %s must be specified" % arg)

        for arg in self.optional_args:
            setattr(self, arg, kwargs.get(arg, self.optional_args[arg]))


class SubstrateBase(object):
    """ Abstract class for substrate at the bottom of the snowpack.
It provides argument handling and calculation of the permittivity constant for soil case.
"""

    def __init__(self, temperature=None, permittivity_model=None):
        """ Build the substrate at the base of the snowpack

        :param temperature: temperature of the base of the snowpack. Can be the effective temperature if the substrate is slightly transparent

        :param permittivity_model: a function that return the permittivity as a function of frequency and temperature. Can also be a numerical value.

        :param **kwargs: other parameters such as roughness_rms, corr_length, Q, N, etc are required or optional depending on the model. See the document of the model.

        """

        self.temperature = temperature
        self.permittivity_model = permittivity_model  # this is a function, so it automatically becomes a method of substrate


    def permittivity(self, frequency):
        """compute the permittivity for the given frequency using permittivity_model. This method returns None when no permittivity model is
        available. This must be handled by the calling code and interpreted suitably."""

        if self.permittivity_model is None:
            return None
        else:
            return self.permittivity_model(frequency, self.temperature)


def substrate_from_interface(interface_cls):
    """this decorator transform an interface class into a substrate class with automatic method"""
    
    def decorator(cls):
        def __init__(self, temperature=None, permittivity_model=None, **kwargs):
            interface_cls.__init__(self, **kwargs)
            SubstrateBase.__init__(self, temperature=temperature, permittivity_model=permittivity_model)

        def specular_reflection_matrix(self, frequency, eps_1, mu1, npol):
            """compute the reflection coefficients for the azimuthal mode m
               and for an array of incidence angles (given by their cosine)
               in medium 1. Medium 2 is where the beam is transmitted.

            :param eps_1: permittivity of the medium where the incident beam is propagating.
            :param eps_2: permittivity of the other medium
            :param mu1: array of cosine of incident angles
            :param npol: number of polarization

            :return: the reflection matrix
"""
            eps_2 = self.permittivity(frequency)
            if eps_2 is None:
                raise SMRTError("No permittivity_model have been given to the substrate '%s'" % str(interface_cls))
            return interface_cls.specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol)

        def emissivity_matrix(self, frequency, eps_1, mu1, npol):
            """compute the transmission coefficients for the azimuthal mode m
               and for an array of incidence angles (given by their cosine)
               in medium 1. Medium 2 is where the beam is transmitted.

            :param eps_1: permittivity of the medium where the incident beam is propagating.
            :param eps_2: permittivity of the other medium
            :param mu1: array of cosine of incident angles
            :param npol: number of polarization

            :return: the transmission matrix
"""
            eps_2 = self.permittivity(frequency)
            if eps_2 is None:
                raise SMRTError("No permittivity_model have been given to the substrate '%s'" % str(interface_cls))
            return interface_cls.coherent_transmission_matrix(self, frequency, eps_1, eps_2, mu1, npol)

        def auto_add(new_method, dependency):
            new_method_name = new_method.__name__
            if not hasattr(cls, new_method_name) and hasattr(interface_cls, dependency):
                #new_method.__doc__ = "This method is autogenerated from an Interface method.\n\n" + \
                #        getattr(interface_cls, new_method_name).__doc__
                attributes[new_method_name] = new_method
            
        attributes = {
                        '__init__': __init__,
                        '__doc__': cls.__doc__,
                        '__module__': cls.__module__,
                    }
        auto_add(emissivity_matrix, 'coherent_transmission_matrix')
        auto_add(specular_reflection_matrix, 'specular_reflection_matrix')

        newcls = type(cls.__name__, (SubstrateBase, interface_cls), attributes)
        newcls.__doc__ = cls.__doc__
        return newcls
    return decorator


# define the Substrate class that is to be derived for object that are not build from Interface
@substrate_from_interface(Interface)
class Substrate:
    pass


def get_substrate_model(substrate_model):
    """return the class corresponding to the substrate model called name. This function imports the correct module if possible and returns the class"""

    return import_class("substrate", substrate_model)


