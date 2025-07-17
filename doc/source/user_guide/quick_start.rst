#########################
Getting started with SMRT
#########################

1.  Get up to date: git status / git pull
2.  Install requirements
3.  (Run tests)
4.  Jupyter notebooks
5.  SMRT imports
6.  Build simple sensor: use AMSR-E single channel
7.  Build 1-layer snowpack (SHS) as simple as possible
8.  Build model (DMRT-QCA, DORT)
9.  Run model and output results
10. Switch DMRT-QCA with IBA and run
11. Turn ipynb into script and run
12. Website help.

How to get SMRT – Option 1
--------------------------

In anaconda / terminal:

.. code:: shell

   pip install git+https://github.com/smrt-model/smrt

to update SMRT
~~~~~~~~~~~~~~

.. code:: shell

   pip install -U git+https://github.com/smrt-model/smrt

How to get SMRT – Option 2
--------------------------

In anaconda / terminal:

.. code:: shell

   git clone https://github.com/smrt-model/smrt

Already have SMRT installed with git clone ? Check it’s the latest version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   git status

to update SMRT
~~~~~~~~~~~~~~

.. code:: shell

   git pull


If it doesn’t say this, then

.. code:: shell

   git pull

Install required packages….
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anaconda:

.. code:: shell


   conda install --yes --file requirements.txt

Bash:

.. code:: shell


   pip install --user -r requirements.txt

Setting the pythonpath to SMRT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anaconda Ghislain!!!?

Bash (edit your ~/.bashrc)

.. code:: shell

   export PYTHONPATH="/mnt/c/Users/melod/CODEREPO/smrt"

Windows: 1. Go to settings, search for ‘edit the system environment
variables’ 2. Click on ‘Environment Variables’ 3. Create new system
variable 4. Name it PYTHONPATH 5. Enter or browse to the path to your
SMRT directory

Python basics
-------------

Start up a jupyter notebook (conda install jupyter / pip install
jupyter)

.. code:: shell


   jupyter notebook

next slide is hello world demo. Remember brackets for python 3


Python basics: list
~~~~~~~~~~~~~~~~~~~

Lists are a useful way of grouping data.

.. code:: ipython3

    # Make a list of numbers
    test = [1,2,3,4]
    
    test




.. parsed-literal::

    [1, 2, 3, 4]



.. code:: ipython3

    # Doesn't always do what you think....
    print(test*2)

.. code:: ipython3

    # Make a list of strings
    mylist = ['a','b','cc','d']

Lists are numbered from 0 !!!!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    print(mylist[0])

.. code:: ipython3

    print(test[3]*2)

Python basics: Plotting
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # First we need to import the matplotlib plotting package
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    # Make a simple graph
    x = [1,2,4]
    y = [10,40,100]
    plt.plot(x,y); # <- This semicolon isn't needed

.. code:: ipython3

    # Make a simple graph
    x = [1,2,4]
    y = [10,40,100]
    plt.plot(x,y,'bo')

Let’s import the modules we need to run SMRT
--------------------------------------------

We’ll need to make the snowpack, set up the model configuration and some
form of sensor.

For now, we’ll use a module that replicates the AMSR-E radiometers
(6.925, 10.65, 18.7, 23.8, 36.5, 89.9 GHz, 55 degree incidence angle).

.. code:: ipython3

    from smrt import make_snowpack, make_model
    from smrt import sensor_list

.. code:: ipython3

    # in case of ModuleNotFoundError, add this before importing smrt
    # import sys
    # sys.path.append('C:/your_path_to_smrt_directory')

Time to make a snowpack!
------------------------

A single layer, deep snowpack….

We need to know the layer thickness, temperature, sticky hard sphere
radius, stickiness and density

All units are SI
~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Define parameters for a one-layer snowpack
    thickness = [10]  # m
    temperature = [265]  # K
    rad = [1e-4]  # m
    rho = [280]  # kg/m3
    stickiness = [0.2]  # no unit
    
    # Make snowpack
    snowpack = make_snowpack(thickness=thickness, 
                             microstructure_model='sticky_hard_spheres',
                             temperature=temperature, radius=rad, density=rho,
                             stickiness=stickiness)


.. code:: ipython3

    snowpack

See inside an object…..
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    dir(snowpack)

.. code:: ipython3

    # A snowpack has a list of layers objects....
    dir(snowpack.layers[0])

.. code:: ipython3

    print(snowpack.layers[0].temperature)

Decide the type of model you want
---------------------------------

For now we will use DMRT-QCA electromagnetic model with the discrete
ordinates solver (DORT)

.. code:: ipython3

    m = make_model('dmrt_qca_shortrange','dort')
    # m = make_model('iba','dort')   # alternative to use iba

One last thing to do before we run the model: specify a sensor
--------------------------------------------------------------

This could be active or passive (see sensor practical) but for now we
use the prebuilt AMSR-E that we have already imported, and we’ll use
just the 37V channel

.. code:: ipython3

    radiometer = sensor_list.amsre('37V')

.. code:: ipython3

    result = m.run(radiometer, snowpack)

.. code:: ipython3

    print(result.TbV(), result.TbH())

.. code:: ipython3

    # Use all amsre channels
    rads = sensor_list.amsre()
    res = m.run(rads, snowpack)
    
    res.TbV()

Sensors in SMRT
===============

**Goal**: - plot the diagram of thermal emission + backscattering
coefficient from a simple snowpack at 13 GHz - plot the diagram of
thermal emission from a simple snowpack at multiple frequencies (e.g. 19
and 37 GHz)

**Learning**: Sensor and Result object

For the snowpack, you can take the following properties: - thickness =
1000 (means semi-infinite) - sticky_hard_spheres microstructure model -
radius = 100e-6 - density = 300 - temperature = 260 - stickiness = 0.15

The following imports are valid for both excercices:

.. code:: ipython3

    import numpy as np
    
    import matplotlib.pyplot as plt
    %matplotlib notebook   
    # use %matplotlib widget if using jupyterlab instead of jupyter notebook
    
    from smrt import make_model, make_snowpack, sensor_list
    from smrt.utils import dB

Emission / backscatter diagram
------------------------------

.. code:: ipython3

    # prepare the snowpack
    radius = 100e-6
    density=300
    temperature = 260
    sp = make_snowpack(thickness=[1000], microstructure_model='sticky_hard_spheres',
                      density=density, radius=radius, temperature=temperature,
                      stickiness=0.15)

.. code:: ipython3

    # prepare a list for theta from 5 to 65 by step of 5
    theta = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    # prepare two sensors (one active, on passive) at 13 GHz
    radiometer = sensor_list.passive(13e9, theta)
    radar  = sensor_list.active(13e9, theta)

.. code:: ipython3

    # prepare the model and run it successively for each sensor
    m = make_model("iba", "dort")
    res_a = m.run(radar, sp)
    res_p = m.run(radiometer, sp)

.. code:: ipython3

    # for plotting two side by side graphs, the best solution is:
    f, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # plot on left graph 
    axs[0].plot(theta, res_a.sigmaVV()) # adapt x and y to your need
    # plot on right graph
    #axs[1].plot(x, y)  # adapt x and y to your need
    
    # to set axis labels:
    axs[0].set_xlabel("Viewing angle")
    # ...

multi-frequency emission diagram
--------------------------------

.. code:: ipython3

    # prepare 1 sensor object with two frequencies

.. code:: ipython3

    # prepare the model and run it

.. code:: ipython3

    # plot the results on a single graph
    # see results documentation for selecting by frequency
    # http://smrt.readthedocs.io/en/latest/smrt.core.result.html

nowpack inputs
===============

**Goal**: - run SMRT with multi-layer snowpack, adjusting various
parameters (like wet snow) - using real data to drive SMRT

**Learning**: make_snowpack

The following imports are valid for both excercices:

.. code:: ipython3

    import numpy as np
    
    import matplotlib.pyplot as plt
    %matplotlib notebook
    
    from smrt import make_model, make_snowpack, sensor_list
    from smrt.utils import dB

Multi-layer snowpack
--------------------

Prepare a snowpack with a few layers. Variable density (e.g. 300, 350,
330, 400 kg m\ :math:`^{-3}`), variable temperature (e.g. -15°C, -10°C,
-5°C, 0°C) same radius (100\ :math:`\mu`\ m) and same stickiness (0.15).
Choose layer thickness (e.g. 10 cm, 30 cm, …). The last layer must be
very thick (we we’ll work with soil later). N.B. if one argument is
given as a scalar while the thickness is an arratyat least another is
given as a list (or array) the scalar value is automatically applied to
all layers.

.. code:: ipython3

    # prepare the multi-layer snowpack
    sp = 

.. code:: ipython3

    # prepare the sensor. Prepare the model and run it. Print or plot the results

.. code:: ipython3

    # Tips: we can draw the snowpack (a recently added function, maybe buggy) as follow:
    from smrt.utils.mpl_plots import plot_snowpack
    
    plt.figure()
    plot_snowpack(sp, show_vars=['density', 'radius'])

Wet snow
--------

The ``make_snowpack`` function can take several optional arguments for
non-default behavior. One of them is “ice_permittivity_model”.
Currently, the defaut formulation is that from Mätzler 1987 for wet
snow, so you can simply add a ‘volumetric_liquid_water’ argument.

.. code:: ipython3

    from smrt import make_snowpack
    # prepare the multi-layer snowpack
    radius = 100e-6
    density=300
    temperature = 260
    sp = make_snowpack(thickness=[0.1, 10],
                       microstructure_model='sticky_hard_spheres',
                       density=density,
                       radius=radius,                   
                       stickiness=0.15,
                       temperature=temperature,
                       volumetric_liquid_water=[0.01, 0])

.. code:: ipython3

    sp.layers[0].permittivity(1, 10e9)

To make explicit the permittivity formulation (which is needed for non
default permittivity). For this: 1) import the function
wetsnow_permittivity defined in the file permittivity/wetsnow.py and 2)
make a snowpack similar to the previous one except set
ice_permittivity_model. This can be list or scalar (yes, Python does
accept list of functions!).

.. code:: ipython3

    
    from smrt.permittivity.wetsnow import wetsnow_permittivity
    from smrt import make_snowpack
    # prepare the multi-layer snowpack
    radius = 100e-6
    density=300
    temperature = 260
    sp = make_snowpack(thickness=[0.1, 10],
                       microstructure_model='sticky_hard_spheres',
                       density=density,
                       radius=radius,                   
                       stickiness=0.15,
                       temperature=temperature,
                       ice_permittivity_model=wetsnow_permittivity,
                       volumetric_liquid_water=[0.01, 0])
    
    sp.layers[0].permittivity(1, 10e9)

.. code:: ipython3

    # prepare the sensor. Prepare the model and run it. Print or plot the results

Read snowpack data
------------------

The manual method
~~~~~~~~~~~~~~~~~

Most of the time, the snowpack is defined in a file or several files.
This does not change the way to run SMRT, only reading the data is
different. A file called “data-domec-sp1-picard-2014.dat” is provided.
Data has been acquired in Snowpit 1 at Dome C in 2012/13 (G. Picard, A.
Royer, L. Arnaud, M. Fily. Influence of meter-scale wind-formed features
on the variability of the microwave brightness temperature around Dome C
in Antarctica, The Cryosphere, 8, 1105-1119, 2014,
doi:10.5194/tc-8-1105-2014). You can open the file with your favorite
editor to see how it looks or (under linux) use the magics of jupyter
notebooks: put in a cell “!cat data-domec-sp1-picard-2014.dat”.

.. code:: ipython3

    thickness, density, radius, temperature = np.loadtxt("data-domec-sp1-picard-2014.dat", skiprows=2, unpack=True, delimiter=" ")
    # check units in the file and you're ready to go.
    # But wait! To check variable from within jupyter notebooks, just enter the variable name
    # at the end of this cell (or another one) and SHIFT+ENTER to see the values.

.. code:: ipython3

    # make snowpack (without stickiness) and so on

.. code:: ipython3

    # (depending on time) you can also try to plot the simulate the impact of a +/-10% on density, either on all layers or just the first one.

Using pandas dataframe
~~~~~~~~~~~~~~~~~~~~~~

If you have time, you can explore the generic function ‘make_medium’ in
inputs/make_medium.py to create a snowpack from a dataframe. This
dataframe must contain a column ‘medium’ which values is ‘snow’ or ‘ice’
(currently) and call directly make_snowpack and make_ice_column. The
other columns contains the attribute normally used by these make\_\*
functions. This feature is useful in many circonstances when the
snowpack parameters is stored on disk or in a database, but makes the
code more obscure than using make_snowpack and siblings, and can be
dangeurous do to collision in column names and attributes.

Practical: Snow microstructure
==============================

The goal of this exercise is to go through the key steps that lie
between a microstructure image and microwave signatures computed from
SMRT

Imports
-------

.. code:: ipython3

    %matplotlib notebook
    
    from IPython.display import HTML, display
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as img
    
    from scipy import fftpack
    import scipy.optimize as opt
    
    from smrt import make_snowpack, make_model, sensor_list


Background
----------

We consider the Passive Active Microwave and Infrared Radiometer (PAMIR)
that was deployed at WFJ, Davos in the 1980s. In May 1984, PAMIR
measured the emission from a snowpack during two melt-refreeze cycles of
the surface during two consecutive nights. Christian Mätzler
characterized the microstructure of the snowpack by thin sections:

For further details cf (Reber et al., “Microwave signatures of snow
crusts: Modelling and measurements”, Int. J. Remote Sen. 8, 1649, 1987)

Visual inspection of an example microstructure
----------------------------------------------

Execute the following cell and look at the image.

.. code:: ipython3

    fig1 = plt.figure()
    fig1.add_subplot()
    filename = 'images/1984May9section1_SEG.png'
    png_image=img.imread(filename)
    plt.imshow(png_image)




.. parsed-literal::

    <IPython.core.display.Javascript object>





.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7f4596d85f28>



Auxiliary functions
-------------------

For convenience several evaluation steps required below are provided as
functions in the following cell. In a first step don’t care about it, we
will later come back to this cell and modify it where necessary. Execute
the following cell.

.. code:: ipython3

    def ice_indicator_function(image_filename):
        """
            read image and convert it to 1,0 indicator function
        """ 
        image=img.imread(image_filename)
        ice_indicator_function = np.asarray(image)
        return ice_indicator_function
    
    
    def ice_volume_fraction(indicator_function):
        """
            compute ice volume fraction from an image indicator function
        """
        return np.mean(indicator_function)
    
    
    def image_size(indicator_function):
        """
            get the size of the image
        """
        return indicator_function.shape    
      
        
            
    
    def ACF1D(acf2d, axis):
        """
            extract the 1D correlation function along a given axis (0 or 1)
        """
        #slc = [slice(None)] * len(acf2d.shape)
        #slc[axis] = slice(0, acf2d.shape[axis])
        #return acf2d[slc]
        nz, nx = acf2d.shape
        if axis == 1:
            return acf2d[0,0:int((nx+1)/2)]
        elif axis == 0:
            return acf2d[0:int((nz+1)/2),0]
        else:
            return "stuss"
        
    def acf1d_fit_exp(r, acf1d, r_max):
        """
            fit the correlation data acf1d for given lags r in the range [0,r_max] 
            to an exponential
            returns:
        """
        
        # set fitrange
        fitrange = (r < r_max)
        # define residual function for least squares fit
        def residual( p, r, acf ):
            C0 = p[0]
            correlation_length = p[1]
            return ( C0*np.exp( -r/correlation_length) - acf )
    
        # initial values for the optimization
        p0 = np.array([0.2,1e-3])
    
        # least square fit in the required range
        p_opt, info = opt.leastsq(residual, p0, args=(r[fitrange],acf1d[fitrange]))
        C0 = p_opt[0]
        correlation_length = p_opt[1]
        acf1d_exp = residual( p_opt, r, 0 )
        
        return acf1d_exp, [C0, correlation_length]
    
    
    def acf1d_fit_ts(r, acf1d, r_max):
        """
            fit the correlation data acf1d for given lags r in the range [0,r_max] 
            to an exponential
        """
        
        # set fitrange
        fitrange = (r < r_max)
        # define residual function for least squares fit
        def residual( p, r, acf ):
            C0 = p[0]
            correlation_length = p[1]
            repeat_distance = p[2]
            return ( C0*np.exp( -r/correlation_length) * np.sinc(2*r/repeat_distance) - acf )
    
        # initial values for the optimization
        p0 = np.array([0.2,1e-3,1e-3])
    
        # least square fit in the required range
        p_opt, info = opt.leastsq(residual, p0, args=(r[fitrange],acf1d[fitrange]))
        C0 = p_opt[0]
        correlation_length = p_opt[1]
        repeat_distance = p_opt[2]
        acf1d_ts = residual( p_opt, r, 0 )
        return acf1d_ts, [C0, correlation_length, repeat_distance]
    
    
    def ACF2D(indicator_function):
        """
            compute the 2D correlation function for the indicator_function of an image
        """
    
        ##################################################
        # replace the following by the correct code
        ##################################################
        f_2 = ice_volume_fraction(indicator_function)
        aux = fftpack.fftn(indicator_function - f_2)
        power_spectrum = np.abs(aux)**2
        acf2d = fftpack.ifftn(power_spectrum)
        nx, nz = indicator_function.shape
        return acf2d.real / (nx*nz)
        
        #return np.zeros_like(indicator_function)
    
    
    def ssa_from_acf_slope(volume_fraction, acf_slope_at_origin):
        """
            compute the ssa from given slope of an autocorrelation function
            C(r) at the origin and the volume fraction.
            This relation is often called Debye relation
        """
        ##################################################
        # replace the following by the correct code
        ##################################################
        rho_ice = 917
        return 4 * acf_slope_at_origin / volume_fraction / rho_ice 


Task 1: Compute the correlation functions for the image
=======================================================

Execute the following cell. You will see a plot which gives nonsense. Go
back to the previous cell and implement the function ``ACF2D`` to
compute the 2D autocorrelation function of the image (5 lines of code
required). When finished, zoom into the image close to the origin and
try to understand.

.. code:: ipython3

    # get the ice indicator function for an example image
    #
    filename = 'images/1984May9section1_SEG.png'
    pixel_size = 0.021e-3 # in mm
    
    indicator_function = ice_indicator_function(filename)
    # get the volume fraction
    volume_fraction = ice_volume_fraction(indicator_function)
    # ACTION REQUIRED HERE
    # get the 2d correlation function
    acf2d = ACF2D(indicator_function)
    
    
    # get the 1d correlation function along an axis
    acf1d_x = ACF1D(acf2d, 1)
    acf1d_z = ACF1D(acf2d, 0)
    
    # get the corresponding lags
    r_x = pixel_size * np.arange(len(acf1d_x))
    r_z = pixel_size * np.arange(len(acf1d_z))
    
    
    # get the fit versions
    r_max = 100 * pixel_size
    acf1d_fit_exp_x, opt_param_exp_x = acf1d_fit_exp(r_x, acf1d_x, r_max)
    print(opt_param_exp_x)
    acf1d_fit_exp_z, opt_param_exp_z = acf1d_fit_exp(r_z, acf1d_z, r_max)
    print(opt_param_exp_z)
    acf1d_fit_ts_x, opt_param_ts_x = acf1d_fit_ts(r_x, acf1d_x, r_max)
    print(opt_param_ts_x)
    acf1d_fit_ts_z, opt_param_ts_z = acf1d_fit_ts(r_z, acf1d_z, r_max)
    print(opt_param_ts_z)
    
    
    
    # plot
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    
    ax2.plot(r_x, acf1d_x, 's', color='b', label='x: meas')
    ax2.plot(r_x, acf1d_fit_exp_x, '-', color='b', label='x: fit EXP')
    ax2.plot(r_x, acf1d_fit_ts_x, ':', color='b', label='x: fit TS')
    
    ax2.plot(r_z, acf1d_z, 'o', color='r', label='z: meas')
    ax2.plot(r_z, acf1d_fit_exp_z, '-', color='r', label='z: fit EXP')
    ax2.plot(r_z, acf1d_fit_ts_z, ':', color='r', label='z: fit TS')
    
    ax2.set_xlim([0, 80*pixel_size])
    ax2.set_xlabel("Lag (mm)")
    ax2.set_ylabel("Correlation functions")
    
    ax2.legend()



.. parsed-literal::

    [0.25272279286023297, 0.00017889630806690145]
    [0.24372885900645125, 0.00022074313938046856]
    [0.23429310202318848, 0.00026560771249732487, 0.0012000836215736443]
    [0.24095589973746762, 0.00023359171888147185, 0.0033000303730074385]



.. parsed-literal::

    <IPython.core.display.Javascript object>



.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f45965ccda0>



Task 2: SSA comparison
======================

Now we have the correlation functions from a fit of the measured data to
an exponential model and to the Teubner-Strey model. The SSA of a
microstructure is related to the slope of the correlation function at
the origin (cf lecture) and we can infer the SSA from the fit parameters
and compare these. Task: Implement the function ``ssa_from_acf_slope``
in the Auxiliary functions cell (one line of code required) and execute
the following cell. Try to understand the differences.

.. code:: ipython3

    ### Check SSA
    
    
    
    SSA_exp_x = ssa_from_acf_slope(volume_fraction, volume_fraction*(1-volume_fraction)/opt_param_exp_x[1])
    SSA_exp_z = ssa_from_acf_slope(volume_fraction, volume_fraction*(1-volume_fraction)/opt_param_exp_z[1])
    SSA_ts_x = ssa_from_acf_slope(volume_fraction, volume_fraction*(1-volume_fraction)/opt_param_ts_x[1])
    SSA_ts_z = ssa_from_acf_slope(volume_fraction, volume_fraction*(1-volume_fraction)/opt_param_ts_z[1])
    
    
    print("SSA from exponential fit in x direction: ", SSA_exp_x, "m^2/kg")
    print("SSA from exponential fit in z direction: ", SSA_exp_z, "m^2/kg")
    print("SSA from Teubner-Strey fit in x direction: ", SSA_ts_x, "m^2/kg")
    print("SSA from Teubner-Strey fit in z direction: ", SSA_ts_z, "m^2/kg")
    



.. parsed-literal::

    SSA from exponential fit in x direction:  15.0902363525 m^2/kg
    SSA from exponential fit in z direction:  12.2295423491 m^2/kg
    SSA from Teubner-Strey fit in x direction:  10.1638146948 m^2/kg
    SSA from Teubner-Strey fit in z direction:  11.5568633351 m^2/kg


Task 3: Brighness temperature comparison
========================================

| Now we analyze how the different correlation functions influence the
  brightness temperature. To this end we adapt the example from
  https://www.smrt-model.science/getstarted.html
| and use the derived parameters to compute the brightness temperature
  for a homogeneous snowpack characterized by the respective correlation
  functions. This is a lazy task, nothing to implement. Try to
  understand the results.

.. code:: ipython3

    # prepare inputs
    thickness = [100]
    temperature = [270]
    density = volume_fraction * 917
    
    # create an "exponential snowpack"
    corr_length = opt_param_exp_x[1]
    snowpack_exp = make_snowpack(thickness=thickness,
                                 microstructure_model="exponential",
                                 density=density,
                                 temperature=temperature,
                                 corr_length=corr_length)
    
    
    # create a "Teubner-Strey snowpack"
    corr_length = opt_param_ts_x[1]
    repeat_distance = opt_param_ts_x[2]
    snowpack_ts = make_snowpack(thickness=thickness,
                                     microstructure_model="teubner_strey",
                                     density=density,
                                     temperature=temperature,
                                     corr_length=corr_length,
                                     repeat_distance=repeat_distance)
    
    # create the sensor
    radiometer = sensor_list.amsre('37V')
    
    # create the model
    m = make_model("iba", "dort")
    
    # run the model
    result_exp = m.run(radiometer, snowpack_exp)
    result_ts = m.run(radiometer, snowpack_ts)
    
    # outputs
    print("Brightness temperature (Exponential): ", result_exp.TbV(), "K")
    print("Brightness temperature (Teubner Strey): ", result_ts.TbV(), "K")



.. parsed-literal::

    Brightness temperature (Exponential):  229.16680394762955 K
    Brightness temperature (Teubner Strey):  246.78928315675998 K


Task 4: Play around
-------------------

Here are some interesting suggestions what you may have a look at: \*
Compare the value of the ``repeat_distance`` parameter with that of the
``corr_length`` paramters in the Teubner-Strey model \* Change the
parameter ``r_max`` in the Task1 cell to ``5*pixel_size`` and check the
impact on the SSA computation and the brightness temperature.

Conclusion
----------

Keep in mind: Microstructure details matter.

Ice in SMRT
===========

**Goal**: - run SMRT with multi-layer snowpack + multi-layer ice column
- understand the difference between first-year and multi-year ice in
SMRT, which has a profound impact on how the electromagnetic medium is
modeled. This is not necessarily as expected from a sea-ice scientist
point of view.

**Learning**: make_ice_column

**Thanks**: to Ludovic Brucker, Nina Mass, and Mai Winstrup for
implementing the sea-ice and salty snow in SMRT.

.. code:: ipython3

    import numpy as np
    
    import matplotlib.pyplot as plt
    %matplotlib notebook
    
    from smrt import make_model, make_snowpack, make_ice_column, sensor_list, PSU
    from smrt.utils import dB


Ice column
----------

The function make_ice_column works as make_snowpack, but with optional
parameters adapted to sea-ice. The only main difference is a first
argument that specifies the type of ice: firstyear, multiyear, or fresh.
We had to make a clear difference between first-year and multi-year
because of a current limitation of IBA and DMRT. They can only compute
scattering for a two-phase medium, so brines and bubbles can not be
(yet) modeled both as scatterers at the same time.

So in SMRT: - first-year ice = brine pockets in a fresh ice background.
It means that brines absorb and scatter the waves. The background is
weakly absorbant. - multi-year ice = air bubbles in a salty ice
background. It means only air bubbles scatter the waves. The brines in
multi-year ice contribute to absorption as in first year ice, but they
do not scatter. - fresh ice = air bubbles in a fresh ice background.
This is the same as multi-year ice but salinity is not considered.

For other details, please read the doc:
https://smrt.readthedocs.io/en/latest/smrt.inputs.make_medium.html#smrt.inputs.make_medium.make_ice_column

NB: salinity is in kg/kg not PSU. But PSU variable can be imported to
make the conversion.

.. code:: ipython3

    # the simplest example
    temperature = 273 - 5
    salinity = 8 * PSU  # ice salinity
    radius = 0.2e-3  # radius of the brines
    
    ic = make_ice_column('firstyear', thickness=[1.0], microstructure_model='sticky_hard_spheres', 
                         temperature=temperature, salinity=salinity, radius=radius)

.. code:: ipython3

    # prepare the sensor. Prepare the model (with IBA) and run it. Print or plot the results


In the previous example, brine fractional volume was estimated from
temperature (see smrt/permittivity/brine.py for details):

.. code:: ipython3

    ic.layers[0].brine_volume_fraction




.. parsed-literal::

    0.25



But it can also be set: change the previous code to set the
brine_volume_fraction to 0.25. Re-run the model and compare the results.


Are the results the same as before or different? Is this expected?

Note that make_ice_column automatically assumes an infinite layer of
water is under the ice. This can be disabled (set
add_water_substrate=False), in which case it is possible to add another
substrate (e.g. soil).

The water parameters can be changed: water_temperature, water_salinity,
water_permittivity_model

To test this, you can make a shallow ice column, use low frequency and
play with water_temperature or water_salinity to see what happens.


Adapt the previous code to make a multi-year ice column. In this case,
porosity (the fractional volume of air bubble) is a required argument,
and brine_volume_fraction is not needed.

read the doc:
https://smrt.readthedocs.io/en/latest/smrt.inputs.make_medium.html#smrt.inputs.make_medium.make_ice_column

Compare the different results between multiyear and first year ice.


If interested, you can also test the ‘fresh’ type to make lake ice.


Snow on ice
-----------

It is likely you’d like to add snow on ice. This can’t be easier:
prepare the two media and use the + operation to stack them: - ic =
make_ice_column(…) - sp = make_snowpack(…) - seaice = sp + ic Then use
seaice as you would use sp.

Prepare a snowpack with a few layers, say with 3 10-cm thick layers,
same density 350 kg m\ :math:`^{-3}`), temperature (e.g. -10°C, -6°C,
-2°C) same correlation length (100\ :math:`\mu`\ m) and same stickiness
(0.15).

.. code:: ipython3

    # prepare the multi-layer snowpack

.. code:: ipython3

    thickness = [10e-2, 10e-2, 10e-2]
    #sp = make_snowpack(thickness, ....)


.. code:: ipython3

    # Prepare the model and run it. Print or plot the results

(time dependent) You can make a sensitivity analysis to snow depth

Ice permittivity in SMRT
~~~~~~~~~~~~~~~~~~~~~~~~

**Goal**:

::

   - run SMRT with different permittivity functions
   - look inside permittivity functions to consider which might be most appropriate
   - compare impact of different permittivity representations

**Learning**: appreciate how fundamental the permittivity choice is

.. code:: ipython3

    # Standard imports
    import numpy as np
    
    import matplotlib.pyplot as plt
    %matplotlib notebook
    
    from smrt import make_model, make_snowpack, make_ice_column, sensor_list, PSU

.. code:: ipython3

    # Again, use the simplest example
    temperature = 273 - 5
    salinity = 8 * PSU  # ice salinity
    radius = 0.2e-3  # radius of the brines
    
    ic = make_ice_column('firstyear', thickness=[1.0], microstructure_model='sticky_hard_spheres', 
                         temperature=temperature, salinity=salinity, radius=radius)

Have a look at the permittivity model used.

.. code:: ipython3

    ic.layers[0].permittivity_model

Why are there two permittivities specified? Neither have a dependence on
salinity - why is this?

Make a plot of permittivity vs temperature


Change the default permittivity model of brine to
seawater_permittivity_klein76. Note you will need to remake the ice
column (cf. 01_seaice_lakeice tutorial). To find the optional argument
required type ‘make_ice_column’ or see the SMRT documentation. You will
need to import the non-default function

.. code:: ipython3

    from smrt.permittivity.saline_water import seawater_permittivity_klein76
    


Calculate the permittivity temperature dependence of this model and
compare with the brine_permittivity_stogryn85 model


Create a sensor, model and look at the impact of permittivity model(s)
on the results. How important is the choice? Note that the way in which
SMRT calculates the effective permittivity of the medium may also matter
- this is part of the electromagnetic model (see Henning’s Lecture).
Saline snow in SMRT
~~~~~~~~~~~~~~~~~~~

Brine is expelled upwards into the snowpack as sea ice is formed,
leading to highly saline layers at the base of the snowpack. This has an
impact on the microwave behaviour on the snowpack e.g. Nandan et
al. (2017) https://doi.org/10.1002/2017GL074506.

Here is a plot of measured snow salinity from four different sites
during the AKROSS project. Note the high salinity near and at the base
of the snowpack. |Measured salinity profiles from AKROSS project|

**Goal**: - add salinity to snowpack - determine sensitivity of results
to snow salinity

**Learning**: understand impact of snow salinity on microwave scattering

.. |Measured salinity profiles from AKROSS project| image:: Salinity_profile.png

.. code:: ipython3

    # Standard imports
    import numpy as np
    
    import matplotlib.pyplot as plt
    %matplotlib notebook
    
    from smrt import make_model, make_snowpack, make_ice_column, sensor_list, PSU

Make a standard snowpack and ice column

.. code:: ipython3

    # sp_pure = ...
    # ic = ...

Make a saline snowpack. Data from the AK2 site in the above graph are:

========= ============ ==============
Depth top Depth bottom Salinity (PSU)
========= ============ ==============
13        10           1.93
10        7            2.27
7         4            1.14
4         1            1.23
0         0            12.72
0         -5           7.39
========= ============ ==============

Add in the salinity to the snowpack with the argument salinity=[…]

.. code:: ipython3

    # sp_saline = ...

Add salty and non-salty snowpacks to the ice column, construct sensor
and model. Run model on the two profiles and compare non-salty and salty
results.


What has happened? Is this expected? How could you fix it?

Hint:

and optional argument ice_permittivity_model

