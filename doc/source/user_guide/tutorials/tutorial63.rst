################################
Including the atmosphere
################################

Table of Contents
-----------------

**Goal**: - Add atmosphere to snowpack - Investigate the sensitivity to
atmosphere parameters.

**Learning**: How to incorporate simple atmosphere into a snowpack.

.. code:: ipython3

    # Import statements
    import numpy as np
    from smrt import make_snowpack, make_model  # Core model functionality

    import matplotlib.pyplot as plt
    %matplotlib inline

Atmosphere
==========

Note: It is likely that how Atmosphere works in SMRT is going to change
to deal with advanced atmosphere models. This may affect this part of
the tutorial in the future.

Import module and define atmosphere. It is possible to include the
downwelling atmospheric contribution (tbdown), and the upwelling
contribution (tbup) and/or the atmospheric transmissivity (trans). The
default values are tbup, tbdown = 0 and transmittivity = 1.

.. code:: ipython3

    from smrt.atmosphere.simple_isotropic_atmosphere import make_atmosphere

    atmos = make_atmosphere(tbdown=30., tbup=6., trans=0.90)

Challenges
~~~~~~~~~~

1. Set up a snowpack / sensor and model configuration
2. Include atmosphere in model run (medium = atmos + snow)
3. Investigate how output changes with atmospheric transmissivity
