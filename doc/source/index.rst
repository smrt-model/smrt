.. SMRT documentation master file, created by
   sphinx-quickstart on Mon Sep 19 14:08:04 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SMRT Documentation
==================================

The SMRT API documentation describes the structure of the package and modules and provides detailed information on the classes and functions. It is not a practical guide for beginners to learn SMRT even though a few examples are sometimes given. We recommend to first read the tutorials <link here> and then use this API documentation as a further step to exploit SMRT in depth. SMRT extensively uses default/optional arguments in functions to provide a simple yet extendable interface. The API documentation is the only valid/up-to-date reference for these default behaviours as it is auto-generated from source. For developers who want to implement new behaviour in SMRT for their own use or for improving SMRT, we recomend to read the developer guidelines <link here> and to contact the authors of the model to discuss about the best/most generic approach to solve your problem. More documentation for improving SMRT will be prepared in the future.

The following package describes all the packages available in SMRT. The :py:mod:`~smrt.inputs` package includes the functions to build the medium and the sensor configuration, it will include in the future any useful functions for inputs from various sources (text file, snowpack model simulations, etc). The :py:mod:`~smrt.permittivity` package provides formulae to compute the permittivity of raw materials such as ice. The :py:mod:`~smrt.microstructure_model` package includes all the representations of the snow micro-structure available. It provides information on the required and optional parameters of each microstrcuture_model. :py:mod:`~smrt.interface` provides the formulation for different types of inter-layer interfaces (such as flat, rugged in the future).


The :py:mod:`~smrt.substrate` package and :py:mod:`~smrt.atmosphere` packages provide the lower and upper boundary conditions of the radiative transfer. Substrate can represent the soil, ice, ocean.
It is worth noting that these modules describe the half-space semi-infinite media under and above the snowpack. It means they have uniform properties and especially temperature which is common practice when the focus is on the snowpack. However, for a proper fully coupled multi-layered soil-snow-atmosphere radiative transfer model, it would be necessary to describe the soil and the atmosphere as layers (exactly as the snowpack is made of snow layers) and to implement :py:mod:`~smrt.emmodel` adequately to the soil and atmosphere.


The :py:mod:`~smrt.emmodel` package includes all the scattering theories available in SMRT (iba, dmrt, independent spheres (Rayleigh), ...). In some case there is an inter-dependence between the choices of micro-structure and of electromagnetic theory. For instance, :py:mod:`~smrt.emmodel.dmrt_shortrange` only works with :py:mod:`~smrt.microstructure_model.sticky_hard_spheres` microstructure (this is inherent to theory) and :py:mod:`~smrt.emmodel.rayleigh` would work with any microstructure model based on spheres (ie. that defines a `radius` parameter).

The :py:mod:`~smrt.rtsolver` package includes the numerical codes that solves the radiative transfer equation.

The :py:mod:`~smrt.core` package is where the SMRT machinery is implemented and especially the most important objects
:py:class:`~smrt.core.sensor.Sensor`, :py:class:`~smrt.core.layer.Layer`, :py:class:`~smrt.core.snowpack.Snowpack`, :py:class:`~smrt.core.model.Model`, etc. It may be useful to understand how these objects work but it is not necessary as most of them (all) are created by helper functions which are much more convenient to use than class constructors. The only exception, which is worth exploring a bit, is :py:class:`~smrt.core.result.Result`. It provides useful methods to extract the result of the radiative calculation.
In general, it is not recommended to modify/extend :py:mod:`~smrt.core` for normal needs. This package does contain any science.

The :py:mod:`~smrt.utils` package provides various useful tools to work with SMRT, but they are not strictly necessary. This package includes wrappers to some off-the-shelf models such as DMRT-QMS, HUT and MEMLS.


.. toctree::
    :titlesonly:
    :maxdepth: 4

    Inputs <smrt.inputs.rst>
    Permittivity <smrt.permittivity.rst>
    Microstructure Model <smrt.microstructure_model.rst>
    Interface <smrt.interface.rst>
    Substrate <smrt.substrate.rst>
    Atmosphere <smrt.atmosphere.rst>

    Electromagnetic Model <smrt.emmodel.rst>
    Radiative Transfer Solver <smrt.rtsolver.rst>

    Core <smrt.core.rst>

    Utilities and tools <smrt.utils.rst>
    Developer Guidelines <developer_guidelines.rst>
    self

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
