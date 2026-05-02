#########################
Sensitivity analysis
#########################

**Goal**: - run sensitivity analysis to show the impact of a given parameter on the SMRT output

**Learning**:

Intuitively running many simulations can be done with a loop and many calls to the SMRT functions. However this is not
the recommended way. SMRT is able to iterate on several parameters of the sensor or the snowpack, and return a unique
result with new coordinates. This is more convenient to work with and allows parallel computation.

A sensor with several frequencies, angles or polarizations is automatically understood by SMRT and the
``Result`` object contains all simulation results as array (i.e. internally as xarray).
The result methods (e.g. TbV()) can return all the values with `xarray.Dataset` `result.TbV()`  and
`xarray.Dataset` `result.TbH()` or can be filtered by frequency, angle or polarization. For instance, to get the
brightness temperature at vertical polarisation for 37 GHz, simply call `result.TbV(frequency=37e9)``.

The same applies when a list of snowpacks is given to the ``run`` method. The ``result`` contains all the computation
results as an array with a dimension `snowpack`, or a customn name if provided.

An even more convenient approach is proposed by using pandas. A pandas DataFrame with a snowpack column can be given to
``run`` and the result is a dataframe with the same column plus the simulation results. This is the most advanced and
powerful way to conduct sensitivity analysis.

In the following, we show these different approaches to conduct sensitivity studies.

Sensitivity with a list of snowpack
===================================

First import the necessary libraries and prepare the sensor and model configuration:

.. code:: ipython3

    import time

    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    %matplotlib notebook

    from smrt import make_model, make_snowpack, sensor_list


The key idea is to build a list of snowpack or a DataFrame. E.g. we want to test the sensitivity of TB’s to the radius.
We first build a list of snowpack with different radius.

.. code:: ipython3

    # prepare the snowpack
    density = 300.0
    radius = np.arange(0.05, 0.5, 0.01) * 1e-3  # from 0.05 mm to 0.5 mm

    # the NAIVE APPROACH:

    snowpack = list()
    for x in radius:
        sp = make_snowpack([1000.0], "sticky_hard_spheres",
                           density=density, temperature=265,
                           radius=x, stickiness=0.15)
        snowpack.append(sp)

In simple cases, it is easier to use “list comprehension”, a nice python feature to create list.

.. code:: ipython3

    # a BETTER APPROACH with list comprehension
    snowpack = [make_snowpack([1000.0], "sticky_hard_spheres",
                              density=density, temperature=265,
                              radius=x, stickiness=0.15) for x in radius]

.. code:: ipython3

    # prepare the sensor and model

    model = make_model("iba", "dort")
    sensor = sensor_list.passive(37e9, 55)

    #run!

Now we have a list of snowpacks, we want to call the model for each snowpack. Here, results is a list of `Results` o
bjects and a loop is again necessary to extract the TB for each snowpack  and plot it. This works but approach is not
recommended.

.. code:: ipython3

    # a NAIVE APPROACH
    # call many times 'run' and get a list of results
    results = [model.run(sensor, sp) for sp in snowpack]

    tbv = [res.TbV() for res in results]
    plt.figure()
    plt.plot(radius, tbv)

Instead, the `run` function can directly take a list of snowpacks and returns a unique result with a new coordinate
named `snowpack`.

.. code:: ipython3

    # a BETTER APPROACH
    # snowpack is a list of Snowpack objects

    results = model.run(sensor, snowpack)

    # results is a Result object not a list

    plt.figure()
    plt.plot(radius, results.TbV())

It is possible to give a custom name and values to the new dimension with `snowpack_dimension` argument.

.. code:: ipython3

    results = model.run(sensor, snowpack, snowpack_dimension=('radius', radius))

    plt.figure()
    plt.plot(results.radius, results.TbV())

The simulations are run in parallel by default, so the computation time is much shorter than the naive approach.
It is possible to disable parallel computation by setting `parallel_computation=False`. It is sometimes easier when
debugging, the error messages are clearer without parallel computation.

  .. code:: ipython3

    results = model.run(sensor, snowpack, snowpack_dimension=('radius', radius), parallel_computation=False)


It is also possible to save the result siumations to disk:

.. code:: ipython3

    results.save("radius-sensitivity.nc")

and later read the results, and get a `Result` object as if the simulations were just run:

    from smrt import open_result

    results = open_result("radius-sensitivity.nc")



Sensitivity with pandas.DataFrame
=================================

Instead of a list of snowpack and providing the dimension name and values, a more concise approach is using
pandas.DataFrame:

.. code:: ipython3

    # here we build a simple DataFrame with the radius. More complex sensitivity analysis with more variables is
    # possible for instance radius and density could co-vary.

    sp = pd.DataFrame({'radius' : np.arange(0.05, 0.5, 0.01) * 1e-3})

    sp['snowpack'] = [make_snowpack([1000.0], "sticky_hard_spheres",
                              density=density, temperature=265,
                              radius=row['radius'], stickiness=0.15) for i, row in sp.iterrows()]

    results = model.run(sensor, sp)

    results

The key step is to add a column named "snowpack" in the DataFrame that contains the `Snowpack`` objects.
While `pandas.DataFrame`` is mainly used with numerical values, it is possible to add any kind of object into the
columns.

This approach is particularly useful when using pandas to read a database of sites, and build the snowpacks directly
from it.

.. code:: ipython3

    # you can even convert the results object to a dataframe

    results = model.run(sensor, sp).to_dataframe()
    # that's it
    results

The `to_dataframe()` method converts the `Result`` object into a dataframe.

It is recommended to use a named sensor (e.g. amsre, smos, …) defined in `smrt.sensor.list`. The sensors define a
channel_map that allows elegant conversion into DataFrame. In this case the columns of the DataFrame are the channels
of the sensor. This is the most convenient way to run multiple simulations and use the results for plotting or computing
statistics.


Recap:
======

The two recommended ways to run sensitivity analysis are:

.. code:: ipython3

    # with List
    snowpack = [make_snowpack([1000.0], "sticky_hard_spheres", density=density, temperature=265, radius=x, stickiness=0.15) for x in radius]

    model = make_model("iba", "dort")
    sensor = sensor_list.amsre(['19', '37'])

    results = model.run(sensor, snowpack, snowpack_dimension=('radius', radius))

    plt.figure()
    plt.plot(results.radius, results.TbV(frequency=19e9), label="19 GHz")
    plt.plot(results.radius, results.TbV(frequency=37e9), label="37 GHz")
    plt.legend()

.. code:: ipython3

    # with DataFrame
    sp = pd.DataFrame({'radius' : np.arange(0.05, 0.5, 0.01) * 1e-3})

    sp['snowpack'] = [make_snowpack([1000.0], "sticky_hard_spheres", density=density, temperature=265, radius=row['radius'],
      stickiness=0.15) for i, row in sp.iterrows()]

    model = make_model("iba", "dort")
    sensor = sensor_list.amsre(['19', '37'])

    results = model.run(sensor, sp, parallel_computation=True).to_dataframe()

    plt.figure()
    plt.plot(results['radius'], results['19V'], label="19 GHz")
    plt.plot(results['radius'], results['37V'], label="37 GHz")
    plt.legend()
