# coding: utf-8

import importlib
import inspect
from functools import lru_cache
from typing import Optional, Type

from smrt.core.error import SMRTError

user_plugin_package = []


def register_package(pkg):
    """
    Register an external package having the same structure as the smrt package, to make available the modules as plugins.

    This is useful for development of independent packages.

    Args:
        pkg: name of the package to register.

    Raises:
        SMRTError: if the package cannot be imported.
    """

    global user_plugin_package

    # check that the package can be imported. It must have an __init__.py
    try:
        importlib.import_module(pkg)
    except ImportError as e:
        raise SMRTError(
            f"The package must be in the the sys.path list and must contain a __init__.py file (even empty). The import error is {str(e)}"
        )

    user_plugin_package.insert(0, pkg)


@lru_cache(maxsize=128)
def import_class(scope: str, modulename: str, classname: Optional[str] = None) -> Type:
    """
    Import the modulename and return either the class named "classname" or the first class defined in the module if
    classname is None.

    Args:
        scope: scope where to search for the module.
        modulename: name of the module to load.
        classname: name of the class to read from the module.
    """

    return _import_object(scope, modulename, classname, _do_import_class)


@lru_cache(maxsize=128)
def import_function(scope: str, modulename: str, funcname: Optional[str] = None) -> Type:
    """
    Import the modulename and return the function named "funcname"
    classname is None.

    Args:
        scope: scope where to search for the module.
        modulename: name of the module to load.
        funcname: name of the function to read from the module.
    """

    return _import_object(scope, modulename, funcname, _do_import_function)


def _import_object(scope: str, modulename: str, objname: Optional[str], do_import_object) -> Type:
    """Import the module and return the object using the importer function."""
    if (".." in modulename) or (modulename[0] == "."):
        raise SMRTError("modulename error. Relative import is not allowed")

    modulename = f"{scope}.{modulename}"

    # add user_directories
    for pkg in user_plugin_package:
        # print(pkg + "." + modulename)
        res = do_import_object(f"{pkg}.{modulename}", objname)
        if res is not None:
            return res

    # the last case, search in the smrt package
    res = do_import_object(f"smrt.{modulename}", objname)

    if res is None:
        if objname is None:
            msg = f"Unable to find the module '{modulename}'."
        else:
            msg = f"Unable to find the module '{modulename}' to import '{objname}'."
        raise SMRTError(msg)

    return res


def _do_import_class(modulename, classname):
    """Import a class from the module name and the class name.

    Args:
        modulename: name of the module to import
        classname: name of the class to load. If None, the first class found in the module is returned.

    Raises:
        SMRTError: if the module can not be found

    """

    try:
        module = importlib.import_module(modulename)
    except ModuleNotFoundError:
        return None

    if classname is None:  # search for the first class defined in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == modulename and not name.startswith(
                "_"
            ):  # the second condition check if the class was defined in this module
                classname = name
                break

    if classname is None:
        raise SMRTError(f"Unable to find a class in the module '{modulename}'")

    # get the class
    cls = getattr(module, classname, None)

    if cls is None:
        raise SMRTError(f"Unable to find the class {classname} in the module '{modulename}'")

    if not inspect.isclass(cls):  # for safety reason
        raise TypeError(f"{cls} in the module '{modulename}' is not a class")

    if cls.__module__ != module.__name__:
        raise TypeError(f"Class {cls} not defined in this module '{modulename}'")

    return cls


def _do_import_function(modulename, funcname):
    """Import a class from the module name and the class name.

    Args:
        modulename: name of the module to import
        classname: name of the class to load. If None, the first class found in the module is returned.

    Raises:
        SMRTError: if the module can not be found

    """

    try:
        module = importlib.import_module(modulename)
    except ModuleNotFoundError:
        return None

    # get the function
    func = getattr(module, funcname, None)

    if func is None:
        raise SMRTError(f"Unable to find the function {funcname} in the module '{modulename}'")

    if not inspect.isfunction(func):  # for safety reason
        raise TypeError(f"{funcname} in the module '{modulename}' is not a function")

    if func.__module__ != module.__name__:
        raise TypeError(f"Function {funcname} not defined in this module '{modulename}'")

    return func
