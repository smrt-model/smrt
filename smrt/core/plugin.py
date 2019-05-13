# coding: utf-8


import importlib
import inspect

from smrt.core.error import SMRTError


def import_class(modulename, classname=None, root=None):
    """import the modulename and return either the class name classname or the first class defined in the module
"""

    if root is not None:
        if "." in modulename:
            raise SMRTError("modulename error. Composed module name is not allowed when root argument is used")
        modulename = root + "." + modulename

    # remove attempt of relative import
    if (".." in modulename) or (modulename[0] == '.'):
        raise SMRTError("modulename error. Relative import is not allowed")

    # import the module
    try:
        module = importlib.import_module(modulename)
    except ImportError as e:
        # TODO: try to import all the modules. Do we want this ??
        raise SMRTError("Unable to find the module '%s' to import the class '%s'. The error is \"%s\"" % (modulename, classname, str(e)))

    if classname is None:  # search for the first class defined in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == modulename:  # the second condition check if the class was defined in this module
                classname = name
                break

    if classname is None:
        raise SMRTError("Unable to find a class in the module '%s'" % modulename)

    # get the class
    return getattr(module, classname)
