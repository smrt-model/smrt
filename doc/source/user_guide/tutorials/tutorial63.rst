###########################
The atmosphere contribution
###########################

Introduction
============

**Goal**: - Add atmosphere to snowpack

The upper boundary limit of the snowpack is called "atmosphere" in SMRT. It represents a single non-scattering layer
with absorption and emission only. This is convenient to model a simple atmosphere over the snowpack, however, more
advanced atmosphere models could be added in the future using the "normal" layering system of SMRT,
thus allowing multiple scattering between atmospheric layers and snow layers.

Several atmosphere models are provided in the `smrt/atmosphere` directory. They are configured and added to the snowpack
by the user using the :py:func:`~smrt.make_atmosphere` function. The RTsolver call this model internally to get 1)
the downwelling radiation, 2) the upwelling radiation, and 3) the transmittance of the layer. This information is used
by the RTsolver to compute the brightness temperature or the backscatter at the sensor level.

Note there is not implementation of the Faraday rotation in the ionosphere currently.

Simple atmosphere
=================

The simplest atmosphere model is an isotropic atmosphere where the transmittance and emission is prescribed.
It can be configured as follows:

.. code:: ipython3

    # Import statements
    import numpy as np
    from smrt import make_atmosphere, make_snowpack, make_model  # Core model functionality

    atmos = make_atmosphere("simple_isotropic_atmosphere", tb_down=25., tb_up=22., transmittance=0.90)

    # make a snowpack
    sp = make_snowpack(thickness, microstruture, ..., atmosphere=atmos)


An alternative to using the atmosphere argument of the `make_snowpack` function is using the addition operation between
the atmosphere and the snowpack.


.. code:: ipython3

    # Import statements
    import numpy as np
    from smrt import make_atmosphere, make_snowpack, make_model  # Core model functionality

    # make a snowpack
    sp = make_snowpack(thickness, microstruture, ...)

    atmos = make_atmosphere("simple_isotropic_atmosphere", tb_down=30., tb_up=27., transmittance=0.90)

    # now add the atmosphere
    sp = atmos + sp

Note that the addition operator is not commutative here, the atmosphere object must come first, the snowpack second.

In general the atmosphere is not isotropic, the transmittance strongly depends on the incidence angle according to the
Beer-Lambert law. This can be modeled with the `simple_atmosphere` model.


Atmosphere model from air temperature and humidity profiles
===========================================================

The absorption and emission of the atmosphere is mainly controlled by the profile of temperature and humidity of the
air in the microwave domain. Several air absorption models exist to perform this complex computation and they can be
coupled with SMRT through the `simple_atmosphere` model. The absorption model computes the absorption of each layer and
using simple radiative transfer calculation outputs the total transmittance and upwelling and downwelling brightness
temperature,  that are used to compute the `simple_atmosphere` model.

Such coupling has been integrated in SMRT for the PyRTlib atmospheric model (Larosa et al. 2024), allowing very
convenient computation of the atmosphere in SMRT. Their model includes several absorption formulations to select
from.

PyRTlib Installation
--------------------

PyRTLib is licensed under the GPL-3.0 License. It can not be distributed along with SMRT, so  and it must be installed
independently of SMRT using:

.. code:: bash

    pip install smrt[pyrtlib]

PyRTlib performs computation using either a climatological profile, or the profile from ERA5 for a given location and
 day and time. ERA5 files are downloaded automatic and cached, optimizing the time of repetitive computations.

A typical simulation with the climatology includes:

.. code:: ipython3

    from smrt import make_atmosphere

    atmos = make_atmosphere('pyrtlib_climatology_atmosphere', profile='Subarctic Summer', absorption_model = 'R20')


while a simulation using ERA5 profiles would include:

.. code:: ipython3

    from smrt.atmosphere.simple_isotropic_atmosphere import make_atmosphere

    atmos = make_atmosphere('pyrtlib_era5_atmosphere', longitude=-75.07, latitude=123., date=datetime(2020, 2, 22, 12),
                            absorption_model = 'R20')

Note that `R20` designates the Rosenkranz model from 2000. Refer to the Larosa et al. 2024  publication: Larosa, S.,
Cimini, D., Gallucci, D., Nilo, S. T., and Romano, F.: PyRTlib: an educational Python-based library for  non-scattering
 atmospheric microwave radiative transfer computations, Geosci. Model Dev., 17, 2053–2076,
https://doi.org/10.5194/gmd-17-2053-2024, 2024.


Recap:
======

- Atmosphere model are for passive microwave use only

- Use `make_atmosphere` to create an atmosphere object, then add the snowpack to it with `sp = atmos + sp`
  or in `make_snowpack`.
- `make_atmosphere("pyrtlib_era5_atmosphere", ...)` is the most advanced way currently available to perform
  an atmospheric computation in SMRT.
- Tip: `simple_isotropic_atmosphere` is also useful to compute the reflectivity of a snowpack, just compare two simulations
  one without atmosphere and one with atmosphere with only tb_down=1K (that is tp_up=0 and transmittance=1).
  The difference in brightness temperature is the reflectivity of the snowpack.
