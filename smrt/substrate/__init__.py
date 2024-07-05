
"""

This package contains different options to represent the substrate, that is the lower boundary conditions of the radiative transfer equation.
This is usually the soil, ice or water but can also be an aluminium plate or an absorber.

To create a substrate, use/implement an helper function such as :py:func:`~smrt.inputs.make_soil.make_soil`. This function is able to 
automatically load a specific soil model .

Examples::

    from smrt import make_soil
    soil = make_soil("soil_wegmuller", "dobson85", moisture=0.2, sand=0.4, clay=0.3, drymatter=1100, roughness_rms=1e-2)

It is recommended to read first the documentation of :py:func:`~smrt.inputs.make_soil.make_soil` and then explore the different types of soil
models.

.. admonition::  **For developers**

    To develop a new substrate formulation, you must add a file in the smrt/substrate directory. The name of the file is used by make_soil
    to build the substrate object.

 """
