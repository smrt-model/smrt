Nadir Altimeter in Low Resolution Mode for snowpack
===================================================

more info, see Larue et al 2021,
https://doi.org/10.1016/j.rse.2021.112534

**Goal**: - Simulate altimeter waveform - Output where contributions are
coming from (surface vs interfaces vs volume) - Observe the influence of
altimeter characteristics and options in the solver

**Learning**:

This tutorial will help you use the following modules -
nadir_lrm_altimetry solver - altimeter_list sensors

First, we import the necessary modules

.. code:: ipython3

    # Standard imports
    from copy import deepcopy
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    from smrt import make_snowpack, make_ice_column, make_model, make_interface
    from smrt.core.globalconstants import C_SPEED
    from smrt.inputs import altimeter_list

Let’s create a one layer snowpack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # add a rough interface at the top
    
    rough_interface = make_interface("geometrical_optics_backscatter", mean_square_slope=0.05)
    
    snowpack = make_snowpack(thickness=[1000], microstructure_model='exponential',
                         density=[350], corr_length=700e-6, temperature=260,
                         surface=rough_interface)


Specify sensor
~~~~~~~~~~~~~~

There are a number of altimeter sensors in altimeter_list. We’ll use
CryoSat-2 in SIN mode.

.. code:: ipython3

    sensor = altimeter_list.sentinel3_sral()

Configure SMRT model
~~~~~~~~~~~~~~~~~~~~

Use the Improved Born Approximation electromagnetic model and the LRM
altimeter solver

.. code:: ipython3

    altimodel = make_model("iba", "nadir_lrm_altimetry")

Run SMRT
~~~~~~~~

.. code:: ipython3

    result = altimodel.run(sensor, snowpack)
    result.sigma()

Observe the content of the result object.

plot the waveform
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    
    ax.plot(result.sigma(), 'k--')
    
    ax.set_xlabel('Gate number', size = 15)
    ax.set_ylabel('Returned power', size = 15)
    plt.tight_layout()

The gate number is the time since recording starts. In general
altimeters try to adjust this starting time in order to keep the leading
edge (the big rise) as close as possible to a prescribeded gate number
(not too early, not too late). In SMRT, the surface corresponds exactly
to a fixed gate number. Here nominal_gate=42 for Sentinel 3. See
parameters in altimeter_list. This has consequences when comparing to
observed waveforms, especially when these waveforms are avaraged (see
later)

Now let convert the gate number into time. In fact SMRT already gives
the time in the result object. See t_gate (in seconds). Redo the
waveform plot as a function of time (in nanosecs) instead of gate number

.. code:: ipython3

    # to change to plot as a function of time
    
    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    
    ax.plot(result.sigma(), 'k--')
    
    ax.set_xlabel('Gate number', size = 15)
    ax.set_ylabel('Returned power', size = 15)
    plt.tight_layout()

Sometimes it is clearer to plot as a function of “apparent” depth.
Convert the t_gate to distance. The equation is :

a_depth = t_gate \* C_SPEED / 2

Note that a_depth is only apparent as the speed of light in the snowpack
is lower than in the vaccum.

.. code:: ipython3

    # to change to plot as a function of apparent depth
    
    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    
    ax.plot(result.sigma(), 'k--')
    
    ax.set_xlabel('Gate number', size = 15)
    ax.set_ylabel('Returned power', size = 15)
    plt.tight_layout()


Separate the contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~

SMRT compute the return from the surface, volume and interfaces
independantly and can return these contributions as follows:

.. code:: ipython3

    altimodel_with_returns = make_model("iba", "nadir_lrm_altimetry", rtsolver_options=dict(return_contributions=True))

.. code:: ipython3

    result_with_returns = altimodel_with_returns.run(sensor, snowpack)

.. code:: ipython3

    #show results in xarray format for more details. See the "contribution" dimension.
    result_with_returns.sigma()

.. code:: ipython3

    # plot all the contributions
    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    
    ax.plot(result_with_returns.t_gate*1e9, result_with_returns.sigma(contribution='total'), 'k-', linewidth=5, label='Total')
    ax.plot(result_with_returns.t_gate*1e9, result_with_returns.sigma(contribution='interfaces'), 'b+-', label='Interfaces')
    ax.plot(result_with_returns.t_gate*1e9, result_with_returns.sigma(contribution='surface'), 'c-', linewidth=3, label='Surface')
    ax.plot(result_with_returns.t_gate*1e9, result_with_returns.sigma(contribution='volume'), 'm--', linewidth=3, label='Volume')
    
    ax.legend()
    ax.set_xlabel('Time (ns)', size = 15)
    ax.set_ylabel('Returned power', size = 15)
    plt.tight_layout()

Here, the snowpack has a rough surface, a volume but no internal
interfaces (see tutorial 02_altimeter_seaice for interfaces)

Further decomposition of the signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To understand the altimetric signal it is convenient to only calculate
the “vertical” component of the echo, as if the altimeter antenna
pattern were infinitely small (like a perfect laser)

.. code:: ipython3

    # This can be done by adjusting the altimeter parameters
    
    from smrt.inputs.altimeter_list import make_multi_channel_altimeter
    def sentinel3_sral_narrow_beam(channel=None):
        config = {
            'Ku': dict(frequency=13.575e9,
                       altitude=814e3,
                       pulse_bandwidth=320e6,
                       nominal_gate=44,
                       ngate=128,
                       beamwidth=0.00001,
                       ),
        }
    
        return make_multi_channel_altimeter(config, channel)

.. code:: ipython3

    # run SMRT with this new altimeter and plot the result
    
    ...

Another (cleaner) way to achieve the same is to use the
“skip_pfs_convolution” options. This stops the calculation before
applying the “Brown” model. See nadir_lrm_altimetry code for available
options.

Note that the nominal_gate is applied with the pfs_convolution, so here
the snowpack surface is at time=0.

.. code:: ipython3

    # create a model with the skip_pfs_convolution option, run and plot
    altimodel_with_returns = make_model("iba", "nadir_lrm_altimetry", rtsolver_options=dict(return_contributions=True,
                                                                                           skip_pfs_convolution=False))
    ...

Simulate more realistic waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The surface is never flat, and this roughness has two consequences:

- influence the power of the surface echo w/r to the volume
  (electromagnetic effect).
- influence the time of return when the height of the topographic within
  the footprint varies by more than the gate equivalent depth.

Both effects have the same origin, but are treated completely
independantly in SMRT. For the former effect see the tutorial
02_altimeter_seaice. For the latter effect, there are two options in
SMRT to simulate it.

The easy one is to add an attribute to the snowpack: sigma_surface which
is the RMS height of the topography. This only controls the second
effect, not the “electromagnetic” roughness that must be controled
independantly.

.. code:: ipython3

    snowpack_topography = deepcopy(snowpack)
    snowpack_topography.sigma_surface = 2  # 2 m RMS height in the footprint.  # the topography is supposed to be normally distributed
    
    results_with_topography = altimodel_with_returns.run(sensor, snowpack_topography)
    results = altimodel_with_returns.run(sensor, snowpack)


.. code:: ipython3

    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    
    ax.plot(result_with_returns.t_gate*1e9, results.sigma(contribution='total'), 'k-', linewidth=5, label='without topography')
    ax.plot(result_with_returns.t_gate*1e9, results_with_topography.sigma(contribution='total'), '-', linewidth=5, label='with topography')
    
    ax.legend()
    ax.set_xlabel('Time (ns)', size = 15)
    ax.set_ylabel('Returned power', size = 15)
    plt.tight_layout()

The second way is to perform the convolution of the signal. This allows
to take into account non-gaussian topography.

The best way is to achieve this is to use “np.convolve” on the waveform


Further investigations
~~~~~~~~~~~~~~~~~~~~~~

1. Change the sensor and/or investigate the effect of the sensor
   parameters.

2. Analysis the sensitivity to the snowpack parameters (e.g. density and
   grain size)

3. Replicate Larue et al. 2021 simulations using data from
   https://github.com/smrt-model/microwave_grain_size_and_polydispersity_paper

or go to the next tutorial to learn more about interfaces, roughness,
more complex environments.


