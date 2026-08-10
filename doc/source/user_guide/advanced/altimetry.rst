Altimetry: Nadir LRM
===================================================

For more information on Low Resolution Mode for nadir altimetry, see Larue et al 2021,
https://doi.org/10.1016/j.rse.2021.112534

**Goals**:

    - Simulate altimetry waveforms
    - Distinguish different contributions (surface, interfaces or volume)
    - Integrate roughness at interfaces

This guide helps you use the
:py:mod:`~smrt.rtsolver.nadir_lrm_altimetry` solver, the :py:mod:`~smrt.interface.geometrical_optics_backscatter` module
and the :py:mod:`~smrt.inputs.altimeter_list` sensors.

Altimetry on snow
-----------------

Let’s first create a simple snowpack for our purpose: one layer and a rough interface at the surface.

.. code:: ipython3

    from smrt import make_snowpack, make_interface

    rough_interface = make_interface("geometrical_optics_backscatter", mean_square_slope=0.05, roughness_rms=0.01)

    snowpack = make_snowpack(thickness=[1000], microstructure_model='exponential',
                         density=[350], corr_length=700e-6, temperature=260,
                         surface=rough_interface)

There are a number of altimeter sensors in altimeter_list. The AltiKa instrument onboard SARAL is used as an example.

.. code:: ipython3

    from smrt.inputs import altimeter_list

    sensor = altimeter_list.saral_altika()

The solver needed is the LRM altimeter solver, which can work with the Improved Born Approximation electromagnetic
model.

.. code:: ipython3

    from smrt import make_model

    altimodel = make_model("iba", "nadir_lrm_altimetry")


The model is then ran as usual. The waveform can be plotted easily:

.. code:: ipython3

    result = altimodel.run(sensor, snowpack)

.. code:: ipython3

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))

    ax.plot(result.sigma(), 'k--')

    ax.set_xlabel('Gate number', size = 15)
    ax.set_ylabel('Returned power', size = 15)
    plt.tight_layout()

The gate number is the time (given in discrete time units) since recording starts. In general altimeters try to adjust
this starting time in order to keep the leading edge (the big rise) as close as possible to a prescribed gate number
(not too early, not too late). In SMRT, the surface corresponds exactly
to a fixed gate number. Here ``nominal_gate=42`` for Sentinel 3. See parameters in
:py:mod:`~smrt.inputs.altimeter_list`. This has consequences when comparing to observed waveforms, especially when
these waveforms are averaged (more on this later).

The waveform can also be plotted as a function of time:

.. code:: ipython3

    # to change to plot as a function of time

    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))

    ax.plot(result.sigma(), result.t_gate() * 1e9, 'k--')

    ax.set_xlabel('Time (ns)', size = 15)
    ax.set_ylabel('Returned power', size = 15)
    plt.tight_layout()

Sometimes it is clearer to plot as a function of “apparent” depth.
To convert ``t_gate`` to a distance, replace ``t_gate`` with:

.. code:: ipython3

    from smrt.core.globalconstants import C_SPEED

    a_depth = t_gate * C_SPEED / 2


Separating different contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The returned power coming from the surface, volume and interfaces is computed independently by SMRT and can be returned
separately.

.. code:: ipython3

    altimodel_with_returns = make_model("iba",
                                        "nadir_lrm_altimetry",
                                        rtsolver_options=dict(return_contributions=True))

    result_with_returns = altimodel_with_returns.run(sensor, snowpack)

.. code:: ipython3

    result_with_returns.sigma()

**Note:** the contribution from internal interfaces is null here as our snowpack does not have any interface.

Further decomposition of the signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To understand the altimetric signal it is convenient to only calculate the strictly vertical component of the echo,
as if the altimeter antenna pattern were infinitely small (like a perfect laser). This can be done by adjusting the
altimeter parameters:

.. code:: ipython3

    def sentinel3_sral_narrow_beam(channel=None):
        config = {'Ku': dict(frequency=13.575e9,
                            altitude=814e3,
                            pulse_bandwidth=320e6,
                            nominal_gate=44,
                            ngate=128,
                            beamwidth=0.00001)}
        return make_multi_channel_altimeter(config, channel)

A cleaner way to achieve the same is to use the ``skip_pfs_convolution`` option. This stops the computation before
applying the Brown77 convolution model. See :py:mod:`~smrt.rtsolver.nadir_lrm_altimetry` code for available
options.

**Note:** the ``nominal_gate`` is applied with the ``pfs_convolution``, so here the snowpack surface is at ``time=0``.

Simulate more realistic waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The surface is never flat, and this roughness has two consequences:

- small scale roughness (typically smaller than the wavelength) influences the power of the surface echo with respect to
the volume (electromagnetic effect).
- large scale roughness (topography) delays the received signal by one or more gate when it is greater than the
gate-equivalent depth.

Both effects have the same origin, but are treated completely independently in SMRT.
More on the electromagnetic roughness at the end of this guide. For the topographic effect, there are two options in
SMRT to simulate it.

1) The easy one is to add a ``sigma_surface`` attribute to the snowpack which
is the RMS height of the topography (considered normally distributed).

.. code:: ipython3

    from copy import deepcopy

    snowpack_topography = deepcopy(snowpack)
    snowpack_topography.sigma_surface = 2  # 2m RMS height in the footprint.

    results_with_topography = altimodel_with_returns.run(sensor, snowpack_topography)

Have a look at a comparison between results considering topography or not.

.. code:: ipython3

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))

    ax.plot(result_with_returns.t_gate*1e9,
            results.sigma(contribution='total'),
            'k-',
            linewidth=5,
            label='without topography')
    ax.plot(result_with_returns.t_gate*1e9,
            results_with_topography.sigma(contribution='total'),
            '-',
            linewidth=5,
            label='with topography')

    ax.legend()
    ax.set_xlabel('Time (ns)', size = 15)
    ax.set_ylabel('Returned power', size = 15)
    plt.tight_layout()

2) The second way to simulate roughness effects can be used to take into account non-gaussian topography. If you have access to a digital elevation model of the surface, you can perform a convolution of the signal over the surface. The best way is to achieve this is to use ``numpy.convolve`` (see Numpy documentation for more information).

Altimetry on sea ice
--------------------

Let’s now create a more complex (two layers) snowpack on top of sea ice. Allow the snow to be saline and use the Scharien permittivity
formulation (the default
permittivity for snow :py:func:`~smrt.ice.permittivity.wetice_permittivity_bohren83` does not depend on
salinity).
Salinity can also specified using PSU unit.
Do not specify the interfaces yet so they are all assumed
to be flat.

.. code:: ipython3

    from smrt.core.globalconstants import PSU
    from smrt import make_ice_column
    from smrt.permittivity.saline_snow import
    saline_snow_permittivity_scharien_with_stogryn95 as ssp

    # specified ice permittivity model of snow
    snow = make_snowpack(thickness=[0.1, 0.2], microstructure_model='exponential',
                         ice_permittivity_model=ssp,density=[300, 350],
                         corr_length=0.5e-4, temperature=260, salinity=[0.001, 0.006] )

    #specfied ice type (firstyear or multiyear)
    ice = make_ice_column(ice_type='firstyear',
                        thickness=[2], temperature=260,
                        microstructure_model='independent_sphere',
                        radius=1e-3,
                        brine_inclusion_shape='spheres',
                        density=910,
                        salinity=8*PSU,
                        add_water_substrate=True)

    medium = snow + ice

There are a number of altimeter sensors to chose from. We’ll use
CryoSat-2 in LRM mode.

.. code:: ipython3

    sensor = altimeter_list.cryosat2_lrm()
    altimodel = make_model("iba", "nadir_lrm_altimetry")

    result = altimodel.run(sensor, medium)

The results can be plotted as above.
The initial sharp rise is called the leading edge, and is often used as
the basis for retracker algorithms to calculate the ice freeboard. The
descending curve is the trailing edge.

Contributions can be computed too. If you plot the signal, you will understand flat interfaces are not realistic.


Rough interface at the snow surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is done the same way as with snow.

.. code:: ipython3

    rough_interface = make_interface("geometrical_optics_backscatter", mean_square_slope=0.03, roughness_rms=0.01)

    rough_snow = make_snowpack(thickness=[0.1, 0.2], microstructure_model='exponential',
                         ice_permittivity_model=ssp,density=[300, 350],
                         corr_length=0.5e-4, temperature=260, salinity=[0.01, 0.06],
                         surface=rough_interface)
    rough_surface_medium = rough_snow + ice

The return is now dominated by the surface. The return from volume
scattering is still of a similar order of magnitude as the previous
smooth surface simulation - you can see this by printing out the largest
amplitude of returns from the surface, there’s just much more from the
rough surface.

If you want to check how the medium is parameterised you can just print
it out and look at the properties:

.. code:: ipython3

    rough_surface_medium

Internal rough surfaces
~~~~~~~~~~~~~~~~~~~~~~~

You can just copy the snowpack and substitute one of the interfaces for
a rough one. A rough interface is already defined and can be
put at the bottom of the medium. Interfaces, as with layers, are
numbered from 0 at the top (-1 just references the last one). Interface
index refer to the top interface of the layer index

.. code:: ipython3

    interface_snow = deepcopy(snow)
    rough_base_medium = interface_snow + ice

    rough_base_medium.interfaces[-1] = rough_interface


Now the altimeter waveform is dominated by scattering from the
interfaces i.e. the snow-sea ice interface.
