Sensitivity analysis
====================

**Goal**: - run sensitivity analysis to show the impact of a given
parameter on the SMRT output

**Learning**:

Intuitively running many simulations can be done with a loop and many
calls to the SMRT functions. But this is not the recommended way, SMRT
makes it more easy (you get a unique ``result`` easier to work with) and
more efficient (SMRT can automatically use parallel computing, possibly
on a High Performance Cluster).

SMRT is indeed able to iterate on several arguments when it is
unambiguous. For instance, a sensor with multiple frequencies, angles or
polarizations is automatically understood. The ``result`` contains all
the values which can be easily accessed with the functions like TbV(),
and can also be filter. E.g. TbV(frequency=37e9)

This is similar when a list of snowpacks is given to ``run``. The
``result`` contains all the computations. The ‘snowpack’ dimension is
automatically added but we can also propose a custom name for this
dimension.

In the recent version, an even more convenient approach is proposed if
you’re using pandas. A pandas DataFrame with a snowpack column can be
given to ``run``. The result once converted to a dataframe contains all
the column of the original DataFrame. This is the most advanced and
powerful way to conduct sensitivity analysis.

In the following, we show different approaches to conduct sensitivity
studies that you can run and then apply to a study case of your choice:
- take the Dome C snowpack and study the sensitivity of TbH 55° to
superficial density - take any snowpack previously defined and
investigated the sensivitiy to liquid_water - etc

.. code:: ipython3

    import time

    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    %matplotlib notebook

    from smrt import make_model, make_snowpack, sensor_list

Build a list of snowpack
------------------------

The key idea is to build a list of snowpack or a DataFrame. E.g. we want
to test the sensitivity of TB’s to the radius. We first build a list of
snowpack with different radius.

.. code:: ipython3

    # prepare the snowpack
    density = 300.0
    radius = np.arange(0.05, 0.5, 0.01) * 1e-3  # from 0 to 0.5 mm

    # the NAIVE APPROACH:

    snowpack = list()
    for x in radius:
        sp = make_snowpack([1000.0], "sticky_hard_spheres",
                           density=density, temperature=265,
                           radius=x, stickiness=0.15)
        snowpack.append(sp)

In simple cases (as this one), it is easier to use “list comprehension”,
a nice python feature to create list.

.. code:: ipython3

    # a BETTER APPROACH with list comprehension
    snowpack = [make_snowpack([1000.0], "sticky_hard_spheres",
                              density=density, temperature=265,
                              radius=x, stickiness=0.15) for x in radius]

    # see an even BETTER APPROACH at the end using pandas.DataFrame

.. code:: ipython3

    # prepare the sensor and model

    model = make_model("iba", "dort")
    sensor = sensor_list.passive(37e9, 55)

    #run!

Now we have a list of snowpacks, we want to call the model for each
snowpack. We can use list comprehension again.

.. code:: ipython3

    # a NAIVE APPROACH
    # call many times 'run' and get a list of results
    results = [model.run(sensor, sp) for sp in snowpack]

    # look at what we get:
    results

This return a list of results. To extract the TB V for each result can
be done with another list comprehension. And then we plot the results.

.. code:: ipython3

    # still the NAIVE APPROACH
    tbv = [res.TbV() for res in results]
    plt.figure()
    plt.plot(radius, tbv)

Nice ? We can do much better because ``Model`` can directly run on a
list of snowpacks. It does not return a list of results, but **a unique
result with a new coordinate** which is much more convenient.

.. code:: ipython3

    # a BETTER APPROACH

    results = model.run(sensor, snowpack, snowpack_dimension=('radius', radius))
    # look at what we get:
    results




.. parsed-literal::

    <smrt.core.result.PassiveResult at 0x7f31e6691eb0>



.. code:: ipython3

    results.coords # look, we have several coordinates, one is call radius




.. parsed-literal::

    Coordinates:
      * theta         (theta) float64 55.0
      * polarization  (polarization) <U1 'V' 'H'
      * radius        (radius) float64 5e-05 6e-05 7e-05 ... 0.00047 0.00048 0.00049
      * frequency     (frequency) float64 1.87e+10 3.65e+10



This is more compact and nicer, ``results`` explicitly show the radius
dimension. Plotting is thus easier:

.. code:: ipython3

    plt.figure()
    plt.plot(results.radius, results.TbV())

And it is easy to save all the result to disk:

.. code:: ipython3

    results.save("radius-sensitivity.nc")

.. code:: ipython3

    # and you get // computation for free, just adding parallel_computation=True

    t0 = time.time()
    results = model.run(sensor, snowpack, snowpack_dimension=('radius', radius))
    print("sequential duration: ", time.time() - t0)

    t0 = time.time()
    results = model.run(sensor, snowpack, snowpack_dimension=('radius', radius), parallel_computation=True)
    print("parallel duration: ", time.time() - t0)


    results

Using pandas.DataFrame
----------------------

.. code:: ipython3

    # here we build a simple DataFrame with the radius. More complex sensitivity analysis with more variables is possible
    # for instance radius and density could co-vary.

    sp = pd.DataFrame({'radius' : np.arange(0.05, 0.5, 0.01) * 1e-3})

    sp['snowpack'] = [make_snowpack([1000.0], "sticky_hard_spheres",
                              density=density, temperature=265,
                              radius=row['radius'], stickiness=0.15) for i, row in sp.iterrows()]

    # show the dataframe
    sp




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>radius</th>
          <th>snowpack</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.00005</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.00006</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.00007</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.00008</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.00009</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.00010</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.00011</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.00012</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.00013</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.00014</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.00015</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.00016</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>12</th>
          <td>0.00017</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>13</th>
          <td>0.00018</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>14</th>
          <td>0.00019</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>15</th>
          <td>0.00020</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>16</th>
          <td>0.00021</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>17</th>
          <td>0.00022</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>18</th>
          <td>0.00023</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>19</th>
          <td>0.00024</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>20</th>
          <td>0.00025</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>21</th>
          <td>0.00026</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>22</th>
          <td>0.00027</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>23</th>
          <td>0.00028</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>24</th>
          <td>0.00029</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>25</th>
          <td>0.00030</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>26</th>
          <td>0.00031</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>27</th>
          <td>0.00032</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>28</th>
          <td>0.00033</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>29</th>
          <td>0.00034</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>30</th>
          <td>0.00035</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>31</th>
          <td>0.00036</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>32</th>
          <td>0.00037</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>33</th>
          <td>0.00038</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>34</th>
          <td>0.00039</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>35</th>
          <td>0.00040</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>36</th>
          <td>0.00041</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>37</th>
          <td>0.00042</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>38</th>
          <td>0.00043</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>39</th>
          <td>0.00044</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>40</th>
          <td>0.00045</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>41</th>
          <td>0.00046</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>42</th>
          <td>0.00047</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>43</th>
          <td>0.00048</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
        <tr>
          <th>44</th>
          <td>0.00049</td>
          <td>Snowpack:       layer                         ...</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    results = model.run(sensor, sp)

    # that's it
    results

.. code:: ipython3

    # you can even convert the results object to a dataframe

    results = model.run(sensor, sp).to_dataframe()
    # that's it
    results

It is recommended to use a named sensor with a channel_map (e.g. amsre,
smos, …) as defined in smrt.sensor.list. In this case the columns of the
DataFrame are the channels of the sensor. It is a very convenient way to
run multiple simulations and use the results for plotting or stats.

.. code:: ipython3

    # try this.

Recap:
------

.. code:: ipython3

    # with List
    snowpack = [make_snowpack([1000.0], "sticky_hard_spheres", density=density, temperature=265, radius=x, stickiness=0.15) for x in radius]

    model = make_model("iba", "dort")
    sensor = sensor_list.passive([19e9, 37e9], 55)

    results = model.run(sensor, snowpack, snowpack_dimension=('radius', radius), parallel_computation=True)

    plt.figure()
    plt.plot(results.radius, results.TbV(frequency=19e9), label="19 GHz")
    plt.plot(results.radius, results.TbV(frequency=37e9), label="37 GHz")
    plt.legend()

.. code:: ipython3

    # with DataFrame
    sp = pd.DataFrame({'radius' : np.arange(0.05, 0.5, 0.01) * 1e-3})

    sp['snowpack'] = [make_snowpack([1000.0], "sticky_hard_spheres",
                              density=density, temperature=265,
                              radius=row['radius'], stickiness=0.15) for i, row in sp.iterrows()]

    model = make_model("iba", "dort")
    sensor = sensor_list.amsre(['19', '37'])

    results = model.run(sensor, sp, parallel_computation=True).to_dataframe()

    plt.figure()
    plt.plot(results['radius'], results['19V'], label="19 GHz")
    plt.plot(results['radius'], results['37V'], label="37 GHz")
    plt.legend()

Do it yourself
--------------

Easy: plot Tb as a function liquid_water_content for a single-layer
snowpack or More invovled: plot a map of Tb(radius, density) using a
single run call (hint: use pd.DataFrame)
