Time-series
===========

**Goal**: Run SMRT on a time-series with varying temperature on the Dome
C snowpack. We propose variations of temperature profile following:

.. math::  T(z, t) = 27 \cos(\omega t - k z) exp(- k z) + (273 - 52) 

with :math:`\omega= 2 \pi/1year` and :math:`k=0.5 m^{-1}`.

**Learning**:

Time-series is frequent, but from a SMRT perspective is not different
from sensitivity study. The idea is to build a list of snowpacks and run
it with snowpack_dimension=(‘time’, time_series). If you know pandas
well, this is even easir with pd.DataFrame, with the benefice of the
power of all the time functions in pandas.

**Tips**:

The computation for a full year (every day) takes a long time. Don’t
forget to use parallel_computation=True in run function.

.. code:: ipython3

    import numpy as np
    
    import matplotlib.pyplot as plt
    %matplotlib notebook
    
    from smrt import make_model, make_snowpack, sensor_list

.. code:: ipython3

    # prepare the snowpack
    thickness, density, radius, tmp = np.loadtxt("data-domec-sp1-picard-2014.dat", skiprows=2, unpack=True, delimiter=" ")
    
    z = np.cumsum(thickness)
    
    def temperature(z, t):
        omega = 2*np.pi/ 365 / 24 / 3600
        k = 0.5
        return 27*np.cos(omega*convert_to_seconds(t) - k*z)* np.exp(-k*z)
    
    def convert_to_seconds(t):
        # this is a bit tricky. It is easier to work with pandas timeseries than numpy timeseries... if you know pandas.
        return (t.astype('datetime64[s]')- np.datetime64('2012-01-01T00:00:00', 's'))/np.timedelta64(1, 's')

.. code:: ipython3

    snowpack = list()
    times = np.arange('2012-01-01', '2013-01-01', 10, dtype='datetime64[D]')

.. code:: ipython3

    plt.figure()
    plt.plot(times, temperature(0, times))
    plt.plot(times, temperature(2, times))

.. code:: ipython3

    # prepare the snowpacks with the different temperature profiles, run and plot the timeseries
    

