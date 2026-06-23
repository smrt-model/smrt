################################
Intensive calculations
################################

**Goal**: - Understand how to run SMRT on a large number of snowpacks in parallel from a single machine to a HPC //

SMRT runs in parallel by default, without any additional configuration, but understanding how this works and how this
can be adjusted is useful for intensive calculations. SMRT has several mechanisms to perform parallel computation, and
they are controled by two arguments of the :py:func:`~smrt.core.Model.run()` function: `parallel_computation` and
`runner`.

By default, `parallel_computation` is set to `auto`. It is the most convenient option for single machine parallelism.
In this mode SMRT first determines how simulations need to be run, that is the product of the number of snowpacks and
sensors configurations to run. If more than one simulation, it selects the `outer` mode which loops over all the
simulations in parallel. The parallelism is handled by the joblib library in separate processes. By default, joblib uses
 all the available cores on the machine to run the simulations in parallel. If there is only one simulation to run
 (1 snowpack and 1 sensor configuration), SMRT selects the `inner` mode which delegates the role of parallelism
to the selected RT solver. Most RTsolvers in SMRT are not parallelized at this stage, but some are, and more will be.

It is also possible to set `parallel_computation` to `outer` or `inner` explicitly. In the `outer` mode,
as mentioned above joblib is used to run each snowpack and sensor configuration in parallel, and this is likely not
efficient if only one simulation is needed. This is why `outer` has no real benefit over the default `auto`.
On the other hand, one may want to set `parallel_computation` to `inner` explicitly, even for many simulations,
if the snowpacks are huge and the RTSolver is using a lot of memory. In such case, the outer parallelism
may cause memory overflow as it starts many RTSolver in parallel, each taking a lot of memory. The inner mode runs
the simulations sequentially, minizing the memory usage, but let the RTSolver performs some internal tasks in parallel.
This is likely less efficient (in speed) than the `outer` mode, but it is safer in terms of memory usage and may
 better fit for some HPC clusters that limit the memory usage per core.

More advanced parallelism settings are controled by the `runner` argument. A runner in SMRT is an object that takes a
list of simulations (list of snowpack and sensor) and run them using some parallelism mechanisms. It is used in
`Model.run` to effectively run the simulations.`. There are several options::
    - `:py:func:~smrt.runner.job_lib.JobLibParallelRunner` is the default runner used by `parallel_computation` in
       outer mode. It uses the joblib library has mentioned earlier. User who want to limit the number of processes or
       control the backend (see joblib documentation) can set
        `runner=JoblibParallelRunner(n_jobs=4, backend='threading')` instead of using `parallel_computation`.
    - `:py:func:~smrt.runner.dask_runner.DaskParallelRunner` uses the well-known Dask library for performance
        computing on cluster. The interest over joblib is to use many nodes on a cluster.
        In such case, it is possible (and recommended) to use `parallel_computation='inner'` to leverage parallelism on
        the RTSolver level (on multi-cores), while letting Dask handle the parallelism on the snowpack/sensor level
        (with nodes).
    - `:py:func:~smrt.runner.celery_runner.CeleryParallelRunner` uses the Celery library, a relatively lightweight
        and robust library for distributed computing. However it has not been tested extensively in SMRT.
    - `:py:func:~smrt.runner.multiprocessing_runner.MultiprocessinglRunner` uses the Pyhton standard library.
        It has not been tested extensively in SMRT, but may be useful when joblib is not available.
    - `:py:func:~smrt.runner.sequential_runner.SequentialRunner` runs the simulations sequentially without any
        parallelism, useful for debugging only or when installing joblib is problematic.

A last control on parallelism in SMRT is performed internally and concerns the RT solvers that leverage LAPACK and
similar libraries. These library are often multi-threaded, and this will use all the available cores on the machine
for their specific tasks. There is benefit to use this multi-threading if there is only one simulation to run, but
otherwise it may overload the machine and slow down the computations. For this reason, some runners (such as joblib)
try to disable the LAPACK multi-threading when there are more than one snowpack or sensor configuration to run.
Conversely, if there is only one simulation to run, it is likely that the numerical libraries will use all the available
cores on the machine for their specific tasks. For user who want to completely disable the multi-threading
of the numerical libraries, it is possible to use `:py:func:smrt.core.lib.set_max_numerical_threads`.










This remaining of this tutorial will help you use parallel computation settings and for the most advanced users
 use a `runner` such as the `dask_runner`.


.. code:: ipython3

    # Standard imports
    import time

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from smrt import make_snowpack, make_ice_column, make_model, make_interface, sensor_list


    %load_ext autoreload
    %autoreload 2

We create large snowpack to evaluate the computational cost of many snowpacks
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

The easiest way to accelerate simulations is to use all the cpus and
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
