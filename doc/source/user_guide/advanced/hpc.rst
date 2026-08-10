######################
Intensive computations
######################

**Goal**: understand how to run SMRT on a large number of snowpacks in parallel on a single machine or a HPC.

SMRT runs the simulations in parallel by default, without any additional configuration, but understanding how
this works and how this can be adjusted is useful for intensive computations. SMRT has several mechanisms to perform
parallel computations, and they are controled by two arguments of the :py:func:`~smrt.core.Model.run()` function:
`parallel_computation` and `runner`.

The `parallel_computation` argument
===================================

By default, `parallel_computation` is set to `"auto"`. It is the most convenient option for single machine parallelism.
In this mode SMRT first determines how many simulations need to be run (number of snowpacks times number of
sensors configurations). If more than one, it selects the `outer` mode which loops over all the
simulations in parallel. This parallelism is handled by the `joblib` library in separate processes. By default, `joblib`
uses all the available cores on the machine to run the simulations in parallel. If there is only one simulation to run,
SMRT selects the `inner` mode which delegates the parallelism to the selected RTsolver. Most RTsolvers in SMRT
are not parallelized at this stage, they run the simulation sequentially, but some are able to run part of
their computation in parallel, and more will be equipped with internal parallelization.

It is also possible to set `parallel_computation` to `inner` explicitly, even for many simulations. This is useful
if the snowpacks are huge and the RTSolver is using a lot of memory. In such case, the outer parallelism
may cause memory overflow as it starts many RTSolver instance in parallel, each taking a lot of memory.
The inner mode runs the simulations sequentially, minizing the memory usage, but let the RTSolver performs some
internal tasks in parallel (if equipped). This is likely less efficient (in speed) than the `outer` mode,
but it is safer in terms of memory usage and may better fit for some HPC clusters that limit the memory usage per core.

At last, it is possible to set to set `parallel_computation` to `outer` explicitly. Compared to `auto`, this avoids
SMRT decide to switch between `outer` and `inner` depending on the number of simulations. There is no obvious benefit to
do this, but it is available for the sake of completeness.

The `runner` argument
=====================

More advanced parallelism settings are controled by the `runner` argument. A "runner" in SMRT is an object that takes a
list of simulations (i.e. list of snowpack and sensor) and runs them using some parallelism mechanisms. It is used in
`Model.run` to effectively run the simulations. There are several runners already developed in SMRT, and more can be
added by users.

- :py:class:`~smrt.runner.joblib_runner.JoblibParallelRunner` is the default runner used by `parallel_computation` in
  `outer` mode (or `auto` mode with more than one simulation). It uses the `joblib` library has mentioned earlier.
  Users who want to limit the number of processes used by joblib or control the joblib backend
  (see `joblib` documentation) can configure the runner explicitly and pass it to `Model.run`. For example, to use
  4 threads instead of all the available cores,   `runner=JoblibParallelRunner(n_jobs=4, backend='threading')`
  instead of using `parallel_computation`.

- :py:class:`~smrt.runner.dask_runner.DaskParallelRunner` uses the well-known Dask library for performance
  computing on cluster. The interest over `joblib` is to use several nodes on a cluster, rather than a single machine or
  node. In such case, it is possible (and recommended) to use `parallel_computation='inner'` to leverage parallelism on
  the RTSolver level (on multi-cores), while letting Dask handle the parallelism on the snowpack/sensor level
  (with nodes).

- :py:class:`~smrt.runner.celery_runner.CeleryParallelRunner` uses the Celery library, a relatively lightweight
  and robust library for distributed computing. It is simple but has not been tested extensively in SMRT.

- :py:class:`~smrt.runner.multiprocessing_runner.MultiprocessinglRunner` uses the Python standard library.
  It has not been tested extensively in SMRT, but may be useful when the external dependency `joblib` is not
  available.

- :py:class:`~smrt.runner.sequential_runner.SequentialRunner` runs the simulations sequentially without any
  parallelism, useful for debugging only or when installing `joblib` is problematic.

A last control on parallelism in SMRT is performed internally and concerns the RT solvers that leverage LAPACK and
similar libraries. These libraries are often multi-threaded, and this will use all the available cores on the machine
for their specific tasks. There is benefit to use this multi-threading if and only if SMRT runs simulations sequentially.
Otherwise, mixing parallelism may overload the machine and slow down the computations. For this reason, some runners
(such as `joblib`) try to disable the LAPACK multi-threading when they are activated and several simulations are to be
run. Conversely, if only one simulation is to be run, it is likely that the numerical libraries will use all the available
cores on the machine for their specific tasks. For user who want to completely disable the multi-threading
of the numerical libraries, it is possible to use :py:func:`smrt.core.lib.set_max_numerical_threads`.


The remainder of this guide illustrates parallel computation settings, and for the most advanced users,
how to use a `runner` such as the `dask_runner`.

Let's create snowpacks with many layers to evaluate the computational cost of many large snowpacks run sequentially:

.. code:: ipython3

    import time
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from smrt import make_snowpack, make_model, sensor_list

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

    for nlayer in list(range(50, 300, 30)) + list(range(300, 1000, 100)):
        print("nlayer:", nlayer)
        sp = create_snowpack(nlayer)
        sps.append(sp)
        t0 = time.time()
        m.run(sensor, sp)  # run single snowpack simulation
        t1 = time.time()
        computations.append({'nlayer': n, 'time': t1 - t0})

    computations = pd.DataFrame(computations)

.. code:: ipython3

    plt.figure()
    plt.plot(computations.nlayer, computations.time)

.. code:: ipython3

    t0 = time.time()
    m.run(sensor, sps, parallel_computation="none")  # force run all simulations sequentially
    t1 = time.time()
    print(f"total computation time: {t1-t0} using SMRT parallelism")

Parallel computation on your machine
====================================

The easiest way to accelerate simulations is to use all the CPU and cores on your machine.
This is what SMRT does by default, using 'parallel_computation="auto"'.  The gain is only for computations at several
frequencies or for many snowpacks.

.. code:: ipython3

    t0 = time.time()
    m.run(sensor, sps, parallel_computation="auto")  # parallel_computation="auto" is not needed, it is the default
    t1 = time.time()
    print(f"total computation time: {t1-t0} using internal SMRT loop")

To deactivate the parallelism, you can use `parallel_computation="none"` or `runner=SequentialRunner()`.

Parallel computation using Dask on an HPC cluster
=================================================

Dask is a Python module for intensive and high memory computations. It works by running one scheduler and one or many
workers on a cluster (or on your local machine for testing). These are just python scripts that are run on the cluster.
This set is often called “a dask cluster” (=the cluster itself + the running scripts). Then, the SMRT simulations are
“pushed” to the scheduler that distributes the simulations on the workers that execute the job in parallel,
and return the results back to SMRT.

The minimal code to use an automatic dask cluster on your local machine is super simple:

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

Due to the network communication overhead, running a simple simulations is usually slower than using
parallel_computation="auto" on a single machine. The main interest is if you have access to an HPC cluster with many
nodes.

.. code:: ipython3

    from dask.distributed import Client
    url = '000.0.0.0:0000'

This url should be replaced by the url of your cluster. The easiest (but not the most performant) way to configure the
network is to use an ssh tunnel e.g. `ssh -N -f HPCCluster -L8799:localhost:8786 sleep 60`
(see the documentation of your cluster for more information)

.. code:: ipython3

    client = Client(url, set_as_default=False, direct_to_workers=False)
    runner=DaskParallelRunner(client)

    t0 = time.time()
    m.run(sensor, sps, runner=runner)
    t1 = time.time()
    print(f"total computation time: {t1-t0} using DASK")

Parallel computation using Ray on an HPC cluster
=================================================

Ray is a Python module for intensive and high memory computations that has several advantages over Dask. However, there
is no Ray runner for SMRT implemented yet. It is planned for the future.
