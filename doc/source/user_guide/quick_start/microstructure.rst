Comparison of microstructure models
===================================

“Grain size” is a major issue when running microwave models. Different
authors have addressed this in different way:

- Tsang’s group tended to use DMRT with fixed stickiness around 0.1, 0.15
  or 0.2 and a grain size from measurements (usually traditional grain
  size measured with hand-lens) until they shift to bi-continuous approach
  (not available in SMRT yet).

- Grenoble+Sherboorke group tended to use DMRT with no stickiness but with
  grain size derived from SSA that is using the classical relationship ``a_opt = 3/SSA/rho_ice``,
  and *scaled by a factor phi*: ``a_dmrt = phi * 3/SSA/rho_ice``.
  They now shifted to the Microwave Grain Size (see last point)

- Mätzler uses the exponential function and, when microstructure images are
  not available, tends to recommend to use scaled Debye relation ``corr_length = X * 3/4 * a_opt* (1-f)`` where f is
  the fractional volume. ``X`` is found to be 0.75.

- Picard et al. (2022) have recently shown that the three approaches give similar results at relatively low frequencies
  (valid 1-100 GHz) provided the microstructure parameters are well adjusted. They deduce a "unified" approach to
  parametrize several microstructure models with three same parameters: Porod length, polydispersity and density. These
  three parameters are rigourously defined and can be measured from microstructure images. The microwave grain size is
  the product of the polydispersity and the Porod length. The unified approach is implemented in SMRT.


In the three first cases, there is one “free” parameter (``stickiness``, scaling ``phi`` or scaling ``X``) that is not
determined from measurements, but is optimized empircally. This parameter is assumed constant for all snowpits and
frequencies to avoid over-fitting. In the latter case, the polydispersity is less well-known, but can be measured from
microstructure images in theory. Typical values are proposed in Picard et al. (2022) for most snow types.

This guide shows that different microstructure gives similar results.

Create a snowpack (as usual) using SHS and ``stickiness=0.1`` (or 0.15 or 0.2), then compute and plot the output.

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

This radius should be 2-3 times larger than the one before, the precise value depends on the stickiness chosen in the
first case.

Repeat the experiment using the ``exponential`` micro-structure and using the scaled debye relationship.
What ``X`` do you get?
