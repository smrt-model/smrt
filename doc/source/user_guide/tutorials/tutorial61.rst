##################################
Adding a substrate
##################################

**Goal**: understand how to add a substrate to simulations in passive and active mode.

Substrate implementation
========================

The substrate is an object containing information namely on the
reflectivity. It can be included as an attribute of a ``snowpack``.

This tutorial shows multiple ways to define a substrate both in passive and active modes.

Passive with direct reflectivity definition
-------------------------------------------

The first passive example demonstrates how to add a substrate with known
reflectivity. This example is for a near perfect absorber (emissivity =
0.98).

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    from smrt import make_snowpack, make_model
    from smrt.inputs.sensor_list import passive

    from smrt.substrate.reflector import make_reflector

The substrate is defined with :py:func:`~smrt.substrate.reflector.make_reflector` and added to the snowpack:

.. code:: ipython3

    substrate = make_reflector(temperature=265, specular_reflection=0.02)
    snow = make_snowpack([1], "exponential", temperature=[265],
                         density=[280], corr_length=[5e-5], substrate=substrate)

Another way to add a substrate is to use the + (or +=) operator, in order for example to keep the same snowpack but easily compare different
representations of the substrate:

.. code:: ipython3

    snow = make_snowpack([1], "exponential", temperature=[265],
                         density=[280], corr_length=[5e-5])
    medium = snow + substrate

You can then have a look at the properties of the
medium - the substrate is listed below the layer properties - then run your model as you would without a substrate:

.. code:: ipython3

    medium

.. code:: ipython3

    m = make_model("iba", "dort")
    rad = passive(21e9, 55)
    m.run(rad, medium).TbV()

Passive with a soil model
-------------------------

Alternatively the soil dielectric constant may be calculated from a
theoretical model. The Wegmuller and Mätzler (1999) soil model is
included in SMRT, other soil models may be contributed by taking a
similar approach. A soil dielectric
constant model is also required. Here we use the Dobson et al., (1985) model.

The substrate is here defined with :py:func:`~smrt.make_soil`:

.. code:: ipython3

    from smrt import make_soil
    soil = make_soil('soil_wegmuller', 'dobson85', temperature=265, roughness_rms=0.25,
                     moisture=0.25, sand=0.01, clay=0.7, drymatter=1300)

Active with prescribed backscatter
----------------------------------

The :py:func:`~smrt.substrate.reflector_backscatter.make_reflector` function used for the active mode is not the same:

.. code:: ipython3

    from smrt.inputs.sensor_list import active
    from smrt.substrate.reflector_backscatter import make_reflector

.. code:: ipython3

    reflector = make_reflector(temperature=265,
                               specular_reflection=0.,
                               backscattering_coefficient={'VV': 0.1, 'HH': 0.1})

The model can the be run as usual:

.. code:: ipython3

    medium = snow + reflector
    scatt = active(13e9, 45)
    model = make_model('iba', 'dort')
    result = model.run(scatt, medium)

You can output the intensity e.g. ``result.sigmaVV()`` directly or in dB with ``result.sigmaVV_dB()``.

Active with backscatter models
------------------------

SMRT currently has two backscatter models implemented: IEM (2 versions)
and Geometric Optics. These are implemented as interfaces, but can be
applied to represent the substrate. Here we’ll use IEM:

.. code:: ipython3

    substrate = make_soil("iem_fung92", "dobson85", temperature=260,
                                                roughness_rms=1e-3,
                                                corr_length=5e-2,
                                                autocorrelation_function="exponential",
                                                moisture=0.25,
                                                sand=0.01,
                                                clay=0.7,
                                                drymatter=1300)


You can also change the bottom (or any!) interface to use the
backscatter model:

.. code:: ipython3

    from smrt.interface.geometrical_optics_backscatter import GeometricalOpticsBackscatter
    from smrt.core.interface import make_interface

    rough_interface = make_interface(GeometricalOpticsBackscatter, mean_square_slope=0.03)
    snow.interfaces[-1] = rough_interface

Look at the snowpack - you can see the interface for the bottom layer
has now changed.

Note that this is the interface at the top of the layer, not the bottom
so it is equivalent here to setting the surface of the snowpack to be
rough. You can also set a rough snow surface by passing the argument
``surface=rough_interface`` when creating the snowpack.
