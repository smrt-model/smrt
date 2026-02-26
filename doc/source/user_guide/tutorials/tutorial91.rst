################################
Intensive calculations
################################

Goal:

::

   Use a HPC cluster to run SMRT in //

Learning:

This tutorial will help you use the following modules

::

   dask_runner

.. code:: ipython3

    # Standard imports
    import time

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from smrt import make_snowpack, make_ice_column, make_model, make_interface, sensor_list


    %load_ext autoreload
    %autoreload 2

We create large snowpack to evaluate the computational cost of snowpacks
with many layers

.. code:: ipython3

    def create_snowpack(nlayer):

        sp = make_snowpack([0.1] * (nlayer - 1) + [1000], "exponential",
                       density=np.maximum(200, np.random.normal(350, 50, nlayer)),
                       corr_length=np.maximum(50e-6, np.random.normal(500e-6, 200e-6, nlayer)),
                       temperature=250)
        return sp

.. code:: ipython3

    sensor = sensor_list.amsre('37V')
    m = make_model("iba", "dort")

.. code:: ipython3

    computations = []
    sps = []

    for n in list(range(50, 300, 30)) + list(range(300, 1000, 100)):
        print("nlayer:", n)
        sp = create_snowpack(n)
        sps.append(sp)
        t0 = time.time()
        # m.run(sensor, sp)  # <-- uncomment this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        t1 = time.time()
        computations.append({'nlayer': n, 'time': t1 - t0})

    computations = pd.DataFrame(computations)

.. code:: ipython3

    plt.figure()
    plt.plot(computations.nlayer, computations.time)

.. code:: ipython3

    t0 = time.time()
    m.run(sensor, sps)
    t1 = time.time()
    print(f"total computation time: {t1-t0} using internal SMRT loop")

Parallel computation on your machine
====================================

the easiest way to accelerate simulations is to use all the cpus and
cores on your machine. Just add “parallel_computation=True” when running
the model.

The gain is only for calculations at several frequencies or for many
snowpacks. Single snowpack at a single frequency is not accelerated, and
indeed will be slower with parallel_computation because in this case
SMRT disable multi-threading in LAPACK.

.. code:: ipython3

    t0 = time.time()
    m.run(sensor, sps, parallel_computation=True)
    t1 = time.time()
    print(f"total computation time: {t1-t0} using internal SMRT loop in //")

Parallel computation using DASK on an HPC cluster
=================================================

Dask is a Python module for intensive and high memory computations. It
works by running one scheduler and one or many workers on a cluster (or
on your local machine for testing). These are just python scripts that
are run on the cluster. This set is often called “a dask cluster” (=the
cluster itself + the running scripts). Then, the smrt simulations are
“pushed” to the scheduler that distributes the simulations on the
workers that execute the job in parallal, and return the results back,
to SMRT.

SMRT abstract most of the boilerplate code to do that.

The minimum code using an automatically a dask cluster on your local
machine is super simple:

.. code:: ipython3

    from dask.distributed import Client
    from smrt.runner.dask_runner import DaskParallelRunner

    client = Client()
    runner=DaskParallelRunner(client)

    t0 = time.time()
    m.run(sensor, sps, runner=runner)
    t1 = time.time()
    print(f"total computation time: {t1-t0} using DASK")

You can monitor the activity on the cluster (if the simulation is long
enough):

http://localhost:8787

Due to the network communication, it is not usually slower than using
parallel_computation=True. The main interest is if you have access to a
big cluster somewhere.

.. code:: ipython3

    from dask.distributed import Client

    url = '127.0.0.1:8799'  # url of your cluster. The easiest way to configure the network is to use ssh tunnel (not the most performant)
    # e.g. ssh -N -f HPCCluster -L8799:localhost:8786 sleep 60


    client = Client(url, set_as_default=False, direct_to_workers=False)
    runner=DaskParallelRunner(client)

    t0 = time.time()
    m.run(sensor, sps, runner=runner)
    t1 = time.time()
    print(f"total computation time: {t1-t0} using DASK")
