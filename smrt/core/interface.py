# coding: utf-8

"""This module  implements the base class for all the substrate models.
To create a substrate, it is recommended to use help functions such as :py:func:`~smrt.inputs.make_soil.make_soil`
rather than the class constructor.

"""

import inspect

from .error import SMRTError
from .plugin import import_class
from smrt.core import lib


def make_interface(inst_class_or_modulename, broadcast=True, **kwargs):
    """return an instance corresponding to the interface model with the provided arguments.

    This function imports the interface module if necessary and
    return an instance of the interface class with the provided arguments in **kwargs.

    :param inst_class_or_modulename: a class, and instance or the name of the python module in smrt/interface
    :param **kwargs: all the arguments required by the interface class
    """

    # import the module
    if inst_class_or_modulename is None:
        from ..interface.flat import Flat  # core should not depend on something defined in interface...
        interface_cls = Flat
    elif isinstance(inst_class_or_modulename, str):
        interface_cls = import_class("interface", inst_class_or_modulename)
    elif inspect.isclass(inst_class_or_modulename):
        interface_cls = inst_class_or_modulename
    elif hasattr(inst_class_or_modulename, "specular_reflection_matrix"):
        # we have an instance... good we can directly return it
        return inst_class_or_modulename
    else:
        raise SMRTError("The interface must be either the name of a module in the smrt.interface directory,"
                        " or a class that implements the interface behavior.")

    if broadcast and kwargs:
        lengths = [len(k) for k in kwargs.values() if lib.is_sequence(k)]
        if lengths:
            return [interface_cls(**lib.get(kwargs, i)) for i in range(max(lengths))]

    return interface_cls(**kwargs)  # create an instance of the interface class


class Interface(object):
    """ Abstract class for interface between layer and substrate at the bottom of the snowpack.
It provides argument handling.
"""
    args = []
    optional_args = {}

    def __init__(self, **kwargs):
        """ Build the interface

        :param **kwargs: parameters such as roughness_rms, corr_length, Q, N, etc are required or optional depending on the model.
        See the document of the model.

        """

        # super().__init__()  # must not be call. This interfers with substrate_from_interface presumably

        for arg in self.args:
            if arg in kwargs:
                setattr(self, arg, kwargs.get(arg))
            else:
                raise SMRTError("Parameter %s must be specified" % arg)

        for arg in self.optional_args:
            setattr(self, arg, kwargs.get(arg, self.optional_args[arg]))


class SubstrateBase(object):
    """ Abstract class for substrate at the bottom of the snowpack.
It provides calculation of the permittivity constant for soil case. Argument handline is delegated to the instance of the interface
"""

    def __init__(self, temperature=None, permittivity_model=None):
        """ Build the substrate at the base of the snowpack

        :param temperature: temperature of the base of the snowpack. Can be the effective temperature if the substrate is slightly transparent

        :param permittivity_model: a function that return the permittivity as a function of frequency and temperature. Can also be a numerical value.

        :param **kwargs: other parameters such as roughness_rms, corr_length, Q, N, etc are required or optional depending on the model. See the document of the model.

        """

        # super().__init__()  # must not be call. This interfers with substrate_from_interface presumably

        self.temperature = temperature
        self.permittivity_model = permittivity_model  # this is a function, so it automatically becomes a method of substrate

    def permittivity(self, frequency):
        """compute the permittivity for the given frequency using permittivity_model. This method returns None when no permittivity model is
        available. This must be handled by the calling code and interpreted suitably."""

        if self.permittivity_model is None:
            return None
        else:
            return self.permittivity_model(frequency, self.temperature)

    def __add__(self, other):

        raise SMRTError("Adding to a substrate is not valid. Only adding a snowpack and a substrate (in that order) is valid")
        # return Snowpack(layers=other.layers,
        #                 interfaces=other.interfaces,
        #                 substrate=other.substrate,
        #                 atmosphere=self)

    def __iadd__(self, other):
        raise SMRTError("Inplace addition with a substrate is not a valid operation.")


def substrate_from_interface(interface_cls):
    """this decorator transform an interface class into a substrate class with automatic method"""

    def decorator(cls):
        def __init__(self, temperature=None, permittivity_model=None, **kwargs):
            SubstrateBase.__init__(self, temperature=temperature, permittivity_model=permittivity_model)
            # create an interface instance
            self.interface_inst = interface_cls(**kwargs)
            # transfer the interface_instance's args and optional args here
            for k in self.interface_inst.args:
                setattr(self, k, getattr(self.interface_inst, k))
            for k in self.interface_inst.optional_args:
                setattr(self, k, getattr(self.interface_inst, k))

        def specular_reflection_matrix(self, frequency, eps_1, mu1, npol):

            eps_2 = self.permittivity(frequency)
            if eps_2 is None:
                raise SMRTError("No permittivity_model have been given to the substrate '%s'" % str(interface_cls))
            return self.interface_inst.specular_reflection_matrix(frequency, eps_1, eps_2, mu1, npol)

        def emissivity_matrix(self, frequency, eps_1, mu1, npol):

            eps_2 = self.permittivity(frequency)
            if eps_2 is None:
                raise SMRTError("No permittivity_model have been given to the substrate '%s'" % str(interface_cls))
            return self.interface_inst.coherent_transmission_matrix(frequency, eps_1, eps_2, mu1, npol)

        def diffuse_reflection_matrix(self, frequency, eps_1, mu_s, mu_i, dphi, npol):

            eps_2 = self.permittivity(frequency)
            if eps_2 is None:
                raise SMRTError("No permittivity_model have been given to the substrate '%s'" % str(interface_cls))
            return self.interface_inst.diffuse_reflection_matrix(frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol)

        def ft_even_diffuse_reflection_matrix(self, frequency, eps_1, mu_s, mu_i, m_max, npol):
            eps_2 = self.permittivity(frequency)
            if eps_2 is None:
                raise SMRTError("No permittivity_model have been given to the substrate '%s'" % str(interface_cls))
            return self.interface_inst.ft_even_diffuse_reflection_matrix(frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol)

        def auto_add(new_method, dependency):
            new_method_name = new_method.__name__
            if not hasattr(cls, new_method_name) and hasattr(interface_cls, dependency):
                # new_method.__doc__ = "This method is autogenerated from an Interface method.\n\n" + \
                #        getattr(interface_cls, new_method_name).__doc__
                attributes[new_method_name] = new_method

        attributes = {
            '__init__': __init__,
            '__doc__': cls.__doc__,
            '__module__': cls.__module__,
            'args': interface_cls.args,  # interface must define args and optional_args
            'optional_args': interface_cls.optional_args,
        }
        auto_add(emissivity_matrix, 'coherent_transmission_matrix')
        auto_add(specular_reflection_matrix, 'specular_reflection_matrix')
        auto_add(ft_even_diffuse_reflection_matrix, 'ft_even_diffuse_reflection_matrix')
        auto_add(diffuse_reflection_matrix, 'diffuse_reflection_matrix')

        newcls = type(cls.__name__, (SubstrateBase, ), attributes)
        newcls.__doc__ = cls.__doc__
        return newcls

    return decorator


# define the Substrate class that is to be derived for object that are not build from Interface
class Substrate(SubstrateBase, Interface):

    def __init__(self, temperature=None, permittivity_model=None, **kwargs):
        SubstrateBase.__init__(self, temperature=temperature, permittivity_model=permittivity_model)
        Interface.__init__(self, **kwargs)


def get_substrate_model(substrate_model):
    """return the class corresponding to the substrate model called name.
    This function imports the correct module if possible and returns the class"""

    return import_class("substrate", substrate_model)
