##################################
Substrate
##################################

Table of Contents
-----------------

**Goal**: - Passive Substrate: specify the reflectivity - Passive
Substrate: use a soil model - Investigate the sensitivity to substrate
parameters.

**Learning**: How to incorporate substrate into a snowpack for passive
simulations.

Substrate
=========

The substrate is an object itself, containing information on the
reflectivity and other parameters if used to calculate reflectivity. The
substrate object then forms part of the snowpack object.

Passive: with reflectivity
--------------------------

The first passive example demonstrates how to add a substrate with known
reflectivity. This example is for a near perfect absorber (emissivity =
0.98). The substrate is defined, then added to the snowpack.

.. code:: ipython3

    # Import statements
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

    from smrt import make_snowpack, make_model
    from smrt.inputs.sensor_list import passive

    # This is the passive reflector
    from smrt.substrate.reflector import make_reflector

.. code:: ipython3

    # Create substrate from known reflectivity
    substrate = make_reflector(temperature=265, specular_reflection=0.02)

.. code:: ipython3

    # Make simple snowpack, including substrate
    snow = make_snowpack([1], "exponential", temperature=[265],
                         density=[280], corr_length=[5e-5], substrate=substrate)

A new way to do this is to use the + operator (also works with +=):

.. code:: ipython3

    snow = make_snowpack([1], "exponential", temperature=[265],
                         density=[280], corr_length=[5e-5])
    medium = snow + substrate

This means you can keep the same snowpack but easily compare different
representations of the substrate. Have a look at the properties of the
medium - the substrate is listed below the layer properties.

.. code:: ipython3

    medium

Make the model, specify a sensor then run the model (feel free to change
values below!)

.. code:: ipython3

    m = make_model("iba", "dort")
    rad = passive(21e9, 55)
    m.run(rad, medium).TbV()

Challenges:
~~~~~~~~~~~

1. Plot a graph of how the brightness temperature varies with
   reflectivity
2. Compare a shallow snowpack with a deep snowpack - what is happening?


Passive: with soil model
------------------------

Alternatively the soil dielectric constant may be calculated from a
theoretical model. The Wegmuller and Mätzler (1999) soil model is
included in SMRT, other soil models may be contributed by taking a
similar approach. This model, however, means that a soil dielectric
constant model is required. Here we use the Dobson et al., (1985) model.

.. code:: ipython3

    from smrt import make_soil
    soil = make_soil('soil_wegmuller', 'dobson85', temperature=265, roughness_rms=0.25,
                     moisture=0.25, sand=0.01, clay=0.7, drymatter=1300)

Challenges:
~~~~~~~~~~~

1. Make a snowpack with a soil substrate
2. How does this compare with the reflector case above?
3. What is the impact of roughness (or other soil parameters)?
4. Try using the soil_qnh model instead
5. What other soil permittivity models could you use?
