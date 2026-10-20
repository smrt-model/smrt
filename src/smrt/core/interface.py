# coding: utf-8

"""This module implements the base class for all the substrate models.
To create a substrate, it is recommended to use help functions such as :py:func:`~smrt.inputs.make_soil.make_soil`
rather than the class constructor.

"""

import inspect
import itertools

import numpy as np

from smrt.core import lib
from smrt.core.error import SMRTError
from smrt.core.globalconstants import FREEZING_POINT
from smrt.core.layer import Layer
from smrt.core.plugin import import_class


def make_interface(inst_class_or_modulename, broadcast=True, **kwargs):
    """Return an instance corresponding to the interface model with the provided arguments.

    This function imports the interface module if necessary and
    returns an instance of the interface class with the provided arguments in \\**kwargs.

    Args:
        inst_class_or_modulename: a class, and instance or the name of the python module in smrt/interface
        broadcast:  (Default value = True)
        **kwargs:
    """
    # import the module
    if inst_class_or_modulename is None:
        from ..interface.flat import (
            Flat,
        )  # core should not depend on something defined in interface...

        interface_cls = Flat
    elif isinstance(inst_class_or_modulename, str):
        interface_cls = import_class("interface", inst_class_or_modulename)
    elif inspect.isclass(inst_class_or_modulename):
        interface_cls = inst_class_or_modulename
    elif hasattr(inst_class_or_modulename, "specular_reflection_matrix"):
        # we have an instance... good we can directly return it
        return inst_class_or_modulename
    else:
        raise SMRTError(
            "The interface must be either the name of a module in the smrt.interface directory,"
            " or a class that implements the interface behavior."
        )

    if broadcast and kwargs:
        lengths = [len(k) for k in kwargs.values() if lib.is_sequence(k)]
        if lengths:
            return [interface_cls(**lib.get(kwargs, i)) for i in range(max(lengths))]

    return interface_cls(**kwargs)  # create an instance of the interface class


class Interface(object):
    """Abstract class for interface between layer and substrate at the bottom of the snowpack.
    It provides argument handling.

    """

    args = []
    optional_args = {}

    def __init__(self, **kwargs):
        """Build the interface.

        Args:
            **kwargs: parameters such as roughness_rms, corr_length, Q, N, etc are required or optional depending on the
                model. See the documentation of the model.

        """
        # super().__init__()  # must not be call. This interfers with substrate_from_interface presumably

        for arg in self.args:
            if arg in kwargs:
                setattr(self, arg, kwargs.get(arg))
            else:
                raise SMRTError(f"Parameter {arg} must be specified")

        for arg in self.optional_args:
            setattr(self, arg, kwargs.get(arg, self.optional_args[arg]))


class SubstrateBase(Layer):
    # a SubstrateBase is a Layer with infinite thickness and fraction volume 1.
    # to become a substrate, it must also be an interface, so it can be used as a substrate in the RT solver.
    def __init__(self, temperature=FREEZING_POINT, permittivity_model=None, **kwargs):
        super().__init__(
            thickness=np.inf,
            frac_volume=1,
            temperature=temperature,
            permittivity_model=[1, permittivity_model],  # layers need two permittivity values
            **kwargs,
        )
        self.microstructure = None  # a substrate has no microstructure, it is a homogeneous layer

    def permittivity(self, frequency):
        # This function is temperature, all the code will be updated to use the normal Layer method with the same name
        return Layer.permittivity(self, 1, frequency)  # call the Layer method, not the Substrate method

    @property
    def frac_volume(self):
        return 1.0

    @frac_volume.setter
    def frac_volume(self, value):
        if value != 1.0:
            raise SMRTError("The fraction volume of a substrate is always 1.0")

    def emissivity_matrix(self, frequency, eps_1, mu1, npol):
        raise DeprecationWarning(
            "The method 'emissivity_matrix' is deprecated. Use 'coherent_transmission_matrix' instead."
        )


class Substrate(SubstrateBase, Interface):
    """A substrate is a layer with infinite thickness and fraction volume 1. It is also an interface, so it can be used
    as a substrate in the RT solver.

    """

    def __init__(self, temperature=FREEZING_POINT, permittivity_model=None, **kwargs):
        SubstrateBase.__init__(self, temperature=temperature, permittivity_model=permittivity_model, **kwargs)
        # call the interface with the optional arguments
        Interface.__init__(self, **kwargs)


def substrate_from_interface(interface_cls):
    """Decorator to transform an interface class into a substrate class with automatic method

    Args:
        interface_cls:
    """

    def decorator(cls):
        def __init__(self, temperature=None, permittivity_model=None, **kwargs):
            Substrate.__init__(self, temperature=temperature, permittivity_model=permittivity_model, **kwargs)
            # create an interface instance
            self.interface_inst = interface_cls(**kwargs)
            # transfer the interface_instance's args and optional args here
            for k in itertools.chain(self.interface_inst.args, self.interface_inst.optional_args):
                setattr(self, k, getattr(self.interface_inst, k))

        def specular_reflection_matrix(self, frequency, eps_1, mu1, npol):
            eps_2 = self.permittivity(frequency)
            if eps_2 is None:
                raise SMRTError(f"No permittivity_model have been given to the substrate '{interface_cls!s}'")
            return self.interface_inst.specular_reflection_matrix(frequency, eps_1, eps_2, mu1, npol)

        def coherent_transmission_matrix(self, frequency, eps_1, mu1, npol):
            eps_2 = self.permittivity(frequency)
            if eps_2 is None:
                raise SMRTError(f"No permittivity_model have been given to the substrate '{interface_cls!s}'")
            return self.interface_inst.coherent_transmission_matrix(frequency, eps_1, eps_2, mu1, npol)

        def diffuse_reflection_matrix(self, frequency, eps_1, mu_s, mu_i, dphi, npol):
            eps_2 = self.permittivity(frequency)
            if eps_2 is None:
                raise SMRTError(f"No permittivity_model have been given to the substrate '{interface_cls!s}'")
            return self.interface_inst.diffuse_reflection_matrix(frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol)

        def ft_even_diffuse_reflection_matrix(self, frequency, eps_1, mu_s, mu_i, m_max, npol):
            eps_2 = self.permittivity(frequency)
            if eps_2 is None:
                raise SMRTError(f"No permittivity_model have been given to the substrate '{interface_cls!s}'")
            return self.interface_inst.ft_even_diffuse_reflection_matrix(
                frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol
            )

        def field_matrix(self, frequency, eps_1, mu1, npol):
            eps_2 = self.permittivity(frequency)
            if eps_2 is None:
                raise SMRTError(f"No permittivity_model have been given to the substrate '{interface_cls!s}'")
            return self.interface_inst.field_matrix(frequency, eps_1, eps_2, mu1, npol)

        def auto_add(new_method, dependency):
            new_method_name = new_method.__name__
            if not hasattr(cls, new_method_name) and hasattr(interface_cls, dependency):
                # new_method.__doc__ = "This method is autogenerated from an Interface method.\n\n" + \
                #        getattr(interface_cls, new_method_name).__doc__
                attributes[new_method_name] = new_method

        attributes = {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "__module__": cls.__module__,
            "args": interface_cls.args,  # interface must define args and optional_args
            "optional_args": interface_cls.optional_args,
        }
        auto_add(coherent_transmission_matrix, "coherent_transmission_matrix")
        auto_add(specular_reflection_matrix, "specular_reflection_matrix")
        auto_add(ft_even_diffuse_reflection_matrix, "ft_even_diffuse_reflection_matrix")
        auto_add(diffuse_reflection_matrix, "diffuse_reflection_matrix")
        auto_add(field_matrix, "field_matrix")

        # does not work due to metaclass conflicts
        # parents = [o in cls.mro() for o in interface_cls.mro() if o not in [interface_cls, object, Interface]]
        # newcls = type(cls.__name__, tuple(parents) + (SubstrateBase,), attributes)

        newcls = type(cls.__name__, (SubstrateBase,), attributes)
        newcls.__doc__ = cls.__doc__
        return newcls

    return decorator


def get_substrate_model(substrate_model):
    """Return the class corresponding to the substrate model called name.

    Args:
        substrate_model:

    Returns:
        This function imports the correct module if possible and returns the class
    """
    return import_class("substrate", substrate_model)
