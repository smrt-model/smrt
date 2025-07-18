Comparison of microstructure model
==================================

**Goal**: Highlight that very different microstructure may give similar
results.

**The long story**:

“Grain size” is a major issue when running microwave models. Different
authors have addressed this in different way:

- Tsang’s group tends to use DMRT with fixed stickiness around 0.1, 0.15
  or 0.2 and a grain size from measurements (usually traditional grain
  size measured with hand-lens)

- Grenoble+Sherboorke group tends use DMRT with no stickiness but with
  grain size derived from SSA that is using the classical relationship
  a_opt = 3/SSA/rho_ice, and *scaled by a factor phi*. It means a_dmrt =
  phi \* 3/SSA/rho_ice. The Dome C dataset provided in this training
  includign scaled grain size with phi=2.3

- Matzler uses Exponential function and, when microstructure images are
  not available, tends to recommend to use scaled Debye relation, ie.
  corr_length = X \* 3/4 a_opt (1-f) where f is the fractional volume. X
  found to be 0.75.

In all cases, there is one “free” parameter (stickiness, scaling phi or
scaling X) that is not determined from measurements, but is optimized.
This parameter is assumed constant for all snowpits, frequencies, … to
avoid over-fitting

**Specific goal**\ \_:

In this excerice, we’ll show that different microstructure gives similar
results: - create a snowpack (as usual) using SHS and stickiness=0.1 or
0.15 or 0.2 - compute and plot the output (e.g. the angular diagram) -
create another snowpack using stickiness=1000 (=not sticky) and adjust
by hand the radius until you get the same results as before. This radius
should be 2-3 times larger than the one before, the precise value
depends on the stickiness chosen in the first case. - repeat the same
using exponential micro-structure and using scaled debye relationship.

.. code:: ipython3

    import numpy as np
    
    import matplotlib.pyplot as plt
    %matplotlib notebook
    
    from smrt import make_model, make_snowpack, sensor_list

.. code:: ipython3

    # prepare the snowpack
    
    thickness = [10]
    density = 350
    temperature = 270
    radius = 100e-6
    stickiness = 0.1
    
    snowpack = make_snowpack(thickness=thickness, microstructure_model='sticky_hard_spheres',
                             radius=radius, density=density, stickiness=stickiness, temperature=temperature)

.. code:: ipython3

    # run the model and plot the results as done in previous practicals


.. code:: ipython3

    # a new snowpack with scaled radius
    phi = 2.5
    
    scaled_snowpack = make_snowpack(thickness=thickness, microstructure_model='sticky_hard_spheres',
                             radius=phi*radius, density=density, stickiness=stickiness, temperature=temperature)
    
    #... continue
