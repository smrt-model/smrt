Comparison of microstructure models
===================================

“Grain size” is a major issue when running microwave models. Different
authors have addressed this in different way:

- Tsang’s group tends to use DMRT with fixed stickiness around 0.1, 0.15
  or 0.2 and a grain size from measurements (usually traditional grain
  size measured with hand-lens)

- Grenoble+Sherboorke group tends use DMRT with no stickiness but with
  grain size derived from SSA that is using the classical relationship
  ``a_opt = 3/SSA/rho_ice``, and *scaled by a factor phi*: ``a_dmrt =
  phi * 3/SSA/rho_ice``.

- Mätzler uses the exponential function and, when microstructure images are
  not available, tends to recommend to use scaled Debye relation
  ``corr_length = X * 3/4 * a_opt* (1-f)`` where f is the fractional volume. ``X``
  is found to be 0.75.

In all cases, there is one “free” parameter (``stickiness``, scaling ``phi`` or
scaling ``X``) that is not determined from measurements, but is optimized.
This parameter is assumed constant for all snowpits and frequencies to
avoid over-fitting

In this excerice, we’ll show that different microstructure gives similar
results.


Create a snowpack (as usual) using SHS and ``stickiness=0.1`` (or
0.15 or 0.2), then compute and plot the output.

.. code:: ipython3

    from smrt import make_snowpack,

    thickness = [10]
    density = 350
    temperature = 270
    radius = 100e-6
    stickiness = 0.1

    snowpack = make_snowpack(thickness=thickness,
                            microstructure_model='sticky_hard_spheres',
                            radius=radius,
                            density=density,
                            stickiness=stickiness,
                            temperature=temperature)

Now create a non-sticky snowpack (e.g. stickiness=1000). Adjust
by hand the radius until you get the same results as before.

.. code:: ipython3

    phi = 1

    scaled_snowpack = make_snowpack(thickness=thickness,
                                    microstructure_model='sticky_hard_spheres',
                                    radius=phi*radius,
                                    density=density,
                                    stickiness=1000,
                                    temperature=temperature)

This radius
should be 2-3 times larger than the one before, the precise value
depends on the stickiness chosen in the first case.

Repeat the experiment
using the ``exponential`` micro-structure and using the scaled debye relationship. What ``X`` do you get?
