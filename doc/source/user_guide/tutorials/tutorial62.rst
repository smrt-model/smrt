################################
Using active substrate
################################

Table of Contents
-----------------

**Goal**: - Active Substrate: specify the reflectivity - Active
Substrate: use a backscatter model - Investigate the sensitivity to
substrate parameters.

**Learning**: How to incorporate substrate into a snowpack for active
simulations.

.. code:: ipython3

    # Import statements
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

    from smrt import make_snowpack, make_model, make_soil  # Core model functionality
    from smrt.inputs.sensor_list import active
    from smrt.utils import dB

    # Note this is different from the passive substrate.reflector
    from smrt.substrate.reflector_backscatter import make_reflector



Active: prescribed backscatter
------------------------------

.. code:: ipython3

    # Define the substrate
    reflector = make_reflector(temperature=265, specular_reflection=0., backscattering_coefficient={'VV': 0.1, 'HH': 0.1})

.. code:: ipython3

    # Need to define an active sensor
    scatt = active(13e9, 45)

Make a snowpack with a substrate

.. code:: ipython3

    # snow = ...
    # medium = ...

Run model and output results in dB

.. code:: ipython3

    model = make_model('iba', 'dort')
    result = model.run(scatt, medium)

You can output the intensity e.g. result.sigmaVV(). This can be
converted to dB with the utils.dB function imported above

.. code:: ipython3

    dB(result.sigmaVV())

or you can output the result in dB directly:

.. code:: ipython3

    result.sigmaVV_dB()

How does the output vary with backscattering coefficients?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Using backscatter models
------------------------

SMRT currently has two backscatter models implemented: IEM (2 versions)
and Geometric Optics. These are implemented as interfaces, but can be
applied to represent the substrate. Here we’ll use IEM

.. code:: ipython3

    substrate = make_soil("iem_fung92", "dobson85", temperature=260,
                                                roughness_rms=1e-3,
                                                corr_length=5e-2,
                                                autocorrelation_function="exponential",
                                                moisture=0.25, sand=0.01, clay=0.7, drymatter=1300)

Make a single layer snowpack, add substrate and run the model

.. code:: ipython3

    #snow = ...

You can also change the bottom (or any!) interface to use the
backscatter model

.. code:: ipython3

    from smrt.interface.geometrical_optics_backscatter import GeometricalOpticsBackscatter
    from smrt.core.interface import make_interface

    rough_interface = make_interface(GeometricalOpticsBackscatter, mean_square_slope=0.03)
    snow.interfaces[-1] = rough_interface

Look at the snowpack - you can see the interface for the bottom layer
has now changed.

.. code:: ipython3

    snow

Note that this is the interface at the top of the layer, not the bottom
so is equivalent here to setting the surface of the snowpack to be
rough. You can also set a rough snow surface by passing the argument
surface=rough_interface when creating the snowpack.

Construct a medium for snow on sea ice - assign a rough interface for
the highest (or single) sea ice layer. This then makes the interface
between the snow and sea ice rough. Run the model. Look at the change
from a smooth snow-sea ice interface.


*Challenges:*

- Make a multilayer snowpack and look at impact of inserting a rough
  interface for each layer in turn.
- Compare IEM and Geometrical Optics. Where might you use one versus the
  other?
