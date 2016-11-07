"""
The :py:mod:`~smrt.core` package contains the SMRT machinery. It provides the infrastructure that provides basic objects and orchestrates the "science" modules in the
other packages (such as :py:mod:`smrt.emmodel` or :py:mod:`smrt.rtsolver`).

Amongst all, we suggest looking at the documentation of the :py:class:`~smrt.core.result.Result` object.

.. admonition::  **For developers**

    We strongly warn against changing anything in this directory. In principle this is not needed because no "science" is present
    and most objects and functions are generic enough to be extendable from outside (without affecting the core definition). Ask
    advice from the authors if you really want to change something here.

"""