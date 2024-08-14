# coding: utf-8

from typing import Optional, Type

import sys
import os
import importlib
import inspect
from functools import lru_cache

from smrt.core.error import SMRTError


user_plugin_package = []


def register_package(pkg):
    global user_plugin_package

    # check that the package can be imported. It must have an __init__.py
    try:
        module = importlib.import_module(pkg)
    except ImportError as e:
        raise SMRTError("The package must be in the the sys.path list and must contain a __init__.py file (even empty). The import error is %s" % str(e))

    user_plugin_package.insert(0, pkg)


@lru_cache(maxsize=128)
def import_class(scope: str, modulename: str, classname: Optional[str]=None) -> Type:
    """Import the modulename and return either the class named "classname" or the first class defined in the module if classname is None.

    :param scope: scope where to search for the module.
    :param modulename: name of the module to load.
    :param classname: name of the class to read from the module.
"""

    if (".." in modulename) or (modulename[0] == '.'):
        raise SMRTError("modulename error. Relative import is not allowed")

    modulename = scope + "." + modulename

    # add user_directories
    for pkg in user_plugin_package:
        # print(pkg + "." + modulename)
        res = do_import_class(pkg + "." + modulename, classname)
        if res is not None:
            return res

    # the last case, search in the smrt package
    res = do_import_class("smrt." + modulename, classname)
    if res is None:
        if classname is None:
            msg = "Unable to find the module '%s'." % modulename
        else:
            msg = "Unable to find the module '%s' to import the class '%s'." % (modulename, classname)
        raise SMRTError(msg)

    return res


def do_import_class(modulename, classname):

    # check the module
    #spec = importlib.util.find_spec(modulename)
    # if spec is None:
    #    return None

    # import the module

    try:
        module = importlib.import_module(modulename)
    except ModuleNotFoundError:
        return None

    if classname is None:  # search for the first class defined in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == modulename and not name.startswith("_"):  # the second condition check if the class was defined in this module
                classname = name
                break

    if classname is None:
        raise SMRTError("Unable to find a class in the module '%s'" % modulename)

    # get the class
    return getattr(module, classname)
