Getting started with SMRT
-------------------------

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

Got python?
-----------

.. figure:: attachment:2018-01-29%20%285%29.png
   :alt: 2018-01-29%20%285%29.png

   2018-01-29%20%285%29.png

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

.. figure:: attachment:gitstatus.PNG
   :alt: gitstatus.PNG

   gitstatus.PNG

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

.. figure:: attachment:jupyter2.PNG
   :alt: jupyter2.PNG

   jupyter2.PNG

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

.. figure:: images/jupyter_image.png
   :alt: alt text

   alt text

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



.. raw:: html

    <img src="data:image/png;base64,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" width="639.2666666666667">




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



.. raw:: html

    <img src="data:image/png;base64,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0Zo1Zv37e6575ZXMZR0dZu++G7x44w2zAQO84ZIllS4TABAzBEBERQAss+RzgC++OE/Diy/2hief7BcIAgCQAwEQUREAy+zZZz3XDRpktnFj9jarV5s9OPcts7o6b7xokdkLL1S2UABAbBAAERUBsMwSCbOjj/ZcN3t21+XJs78DBpht+vql3nDAALPRo83a2ipfMACg5hEAERUBsALmz/dcN3x49kw3ZYqPCf3H375jtttu3njkSH+uHAAAIQRAREUArIAdO8z22cdz3QMPdF2+fXvai1mzvOGRR2YfQBAA0OcRABEVAbBCbr7Zc92ECXkabtxoNmSIN164sCK1AQDihQCIqAiAFbJ+feoej6VLs7fZts3sxz82az7/Jm84ZozZggW57x4BAPRJBEBERQCsoPp6z3XTp2dfPnOmLz/lmC2W2HPP1ODQf//3lS0UAFDTCICIigBYQStWpG7yXbu26/J168wOP9wP+nX+6Mepx8PNmVP5YgEANYsAiKgIgBU2aZLnuquvzr78L2NAf/CB2d57e+O7765YfQCA2kcARFQEwAp75BHPdMOG+TV/3br9dm+8//6hW4VDOjrMmprMGhv9a0dHD1YMAKg1BEBERQCssI4Os4MO6v7AXiJh9sQTZmd8cbslhu/vjWfNyn4qeNEiD4jJ6wWTgXHRovL+IACAqiEAIioCYBXMm+c5bezY7I/93b7dB42WzP7z7Lszw93vfpdquGiRXyOYvjx53WC/foRAAOilCICIigBYBa2tZoMHe1Z76qnsbX7xC7MrrjDbsK7NbNQobzxunD87zswPJYaP/IVD4IgRnA4GgF6IAIioCIBV8t3vek6bMqWAxslnyQ0bZrZli89rasod/tKnpqYy/hQAgGogACIqAmCVvPFG6uztqlV5Gnd0WOLQQ73xTTf5vMbGwgJgY2PZfxYAQGURABEVAbCKTj/dM9qMGbnbtLT4ANL3/c1Cb/zRj5pddVXq2XIcAQSAPocAiKgIgFX0m994Rtt9d7NNm7K3+fWvvc2guk5rP2J8Kth98pN+p0i2m0C4BhAAejUCIKIiAFZRImH2V3/leW3u3NxtZs40W7bMzB591Bv37292771mDz+cuuOXu4ABoM8gACIqAmCV/cu/eGY74ACz9vY8jRMJswkTfIVLL/V52cYBHDGC8AcAvRgBEFERAKts+3az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" width="640">




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

.. figure:: Permittivity.png
   :alt: Spider graph of SMRT permittivity functions

   Spider graph of SMRT permittivity functions

and optional argument ice_permittivity_model

