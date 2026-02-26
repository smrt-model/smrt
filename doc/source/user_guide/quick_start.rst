#########################
Getting started with SMRT
#########################

This tutorial is aimed at a complete beginner with no experience of running SMRT. By following the code, you will run SMRT
in passive mode for a simple, dry, 2-layer snowpack by learning how to:

#. Generate inputs for the model
#. Configure the model
#. Run the model
#. Look at the output


Model inputs
------------
In SMRT, the *medium* consists of a stack of horizontal layers representing snow or ice, with rough or smooth *interfaces* between layers.
The *substrate* and *atmosphere* provide the lower and upper boundary conditions of the radiative transfer and describe the half-space
semi-infinite media under and above the snowpack. It means they have uniform properties and especially temperature, which is common
practice when the focus is on the snowpack. We'll assume no atmospheric effects for now and will consider a semi-infinite snowpack i.e.
one that it so deep the radiation does not penetrate to the bottom of the snowpack. This means we do not have to specify a substrate just yet.

First we will define the properties of each layer: thickness, temperature, density and microstructure. The easiest way to
do this is to specify each property in the form of a list. The length of the list should equal the number of layers in
the snowpack. Here, for a 2-layer snowpack the lists are of length 2.


.. important:: Note that in SMRT:

    * Layers are numbered from the top i.e. the first number in the list is the top layer of the snowpack.
    * All units are S.I. If you have unexpected results, check the input units first!
    * Thickness must always be specified as a list: SMRT determines the number of layers from the length of this list. Other inputs can be scalar to apply to all layers.

.. code:: ipython3

    thickness_list = [0.1, 1000]
    density_list = [200, 350]
    temperature_list = [265, 250]
    corr_length_list = [0.1e-3, 0.3e-3]

The top 10cm layer of the snowpack has a density of 200 kg m\ :sup:`-3`, temperature 265K and correlation length 0.1mm.
The bottom 1km of the snowpack has a density of 350 kg m\ :sup:`-3`, temperature of 250K and correlation length 0.3mm.
This isn't a physically realistic snowpack but just illustrates how to use SMRT. To add a layer to the snowpack, simply
extend the length of the lists. Here we have used correlation length to describe the snow microstructure, and we will
assume an exponential microstructure model (more on that later). Other microstructure models require alternative parameters
e.g. sticky hard spheres will need radius rather than correlation length. These inputs are the bare minimum needed to run SMRT.
See the API for other inputs e.g. liquid water, salinity that can also be added into the make_snowpack function.

SMRT requires a snowpack object to run the model. There is a helper function to make it easy to transform parameters into the snowpack object,
but this needs to be imported. There are other helper functions for e.g. ice, substrate and atmosphere depending on your needs.
Import the ```make_snowpack``` function with

.. code:: ipython3

    from smrt import make_snowpack

then make the snowpack with

.. code:: ipython3

    my_snowpack = make_snowpack(thickness=thickness_list,
                            density=density_list,
                            temperature=temperature_list,
                            microstructure_model='exponential',
                            corr_length=corr_length_list)

For a quick check to visualise the layer properties of the snowpack, you can do

.. code:: ipython3

    from smrt.utils.mpl_plots import plot_snowpack
    plot_snowpack(my_snowpack, show_vars=['density'])

Note that in this case, the top layer is so thin compared with the bottom layer that you will not be able to see it.
Try changing the thickness of the layers to be similar: they will be much clearer with the
```plot_snowpack``` function. Remember to reset the lower layer to be very deep.

The other *input* needed to run the model is the sensor, containing information related to the observation technique i.e.
instrument characteristics. There are many pre-built sensors (e.g. AMSR-E) that can simply be imported from smrt.inputs.sensor_list, but here
we will specify our own simple passive sensor covering two frequencies (19 and 37 GHz) operating at a single incidence
angle of 50\ :sup:`o`.

.. code:: ipython3

    from smrt.inputs import sensor_list
    frequency = [19e9, 37e9]
    incidence_angle = 50
    radiometer = sensor_list.passive(frequency, incidence_angle)

.. tip::
    Simulations can be performed for multiple frequencies and/or incidence angles by specifying these as lists.


Model Configuration
-------------------
In order to run SMRT, you need to specify which electromagnetic model and radiative transfer solver to use. The electromagnetic
model uses the physical properties of the snowpack to calculate electromagnetic properties i.e. phase function
(angular distribution of scattered radiation) and scattering and absorption coefficients. Different electromagnetic models have
different underlying assumptions and simplifications, so it is important to understand these limitations when making a choice.
One example of this is the connection to microstructure model e.g. DMRT-QCA is fundamentally derived from a sticky hard sphere
microstructure model, so cannot be used with any other. For these simulations we will use the Improved Born Approximation (IBA),
which has no constraints on the microstructure model.

Numerous options to solve the radiative transfer equations are available in SMRT. The most generic and applicable is the Discrete
Ordinates Radiative Transfer (DORT), which we will use here, is suitable for both passive and active simulations. There is no need to
specify which: SMRT will determine this from the *sensor* characteristics. For simulation of radar echo waveforms, different radiative
transfer solvers are needed as these solve the time-dependent radiative transfer equation. Alternative radiative transfer solvers
may be used to reduced computation time, or determine the proportion of radiation that has undergone singular versus multiple
scattering, but here we will simply use DORT: it simulates scattering in multiple streams that are connected between layers via
Snell's Law. Again, we use a helper function  ```make_model``` to build the model

.. code:: ipython3

    from smrt import make_model
    model = make_model('iba', 'dort')

SMRT is now ready to use!


Run the model
--------------

Once the inputs and model configuration are specified, running SMRT is very simple:

.. code:: ipython3

    result = model.run(radiometer, my_snowpack)

.. note::

    Order of inputs matters: sensor first, then snowpack unless you specify the arguments explicitly e.g. ```model.run(snowpack=my_snowpack, sensor=radiometer)```


Look at the results
--------------------

The output of the model is a *Result* object. Brightness temperature values can be output with a call to the TbV() and TbH()
functions e.g.

.. code:: ipython3

    result.TbV()

This displays the brightness temperature at vertical polarisation in xarray form (or scalar if a single frequency and incidence angle is specified). If you
want just a single value you can specify how to slice the data along a particular dimension by passing an argument into the TbV function e.g.

. code:: ipython3

    result.TbV(frequency=19e9)
