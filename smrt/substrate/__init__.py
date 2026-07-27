"""Contains different options to represent the substrate, that is, the lower boundary conditions of the radiative
transfer equation.

This is usually the soil, ice, or water but can also be an aluminium plate or an absorber.

To create a substrate, use or implement a helper function such as smrt.inputs.make_soil.make_soil_substrate.
This function is able to automatically load a specific soil model.

Examples::

    from smrt import make_soil_substrate
    soil = make_soil_substrate("soil_wegmuller", "dobson85_peplinski95", moisture=0.2, sand=0.4, clay=0.3,
                    drymatter=1100, roughness_rms=1e-2)

It is recommended to first read the documentation of smrt.inputs.make_soil.make_soil_substrate and then explore the
different types of soil models.

.. admonition::  **For developers**

    To develop a new substrate formulation, you must add a file in the smrt/substrate directory. The name of the file is
    used by make_soil_substrate to build the substrate object.
"""
