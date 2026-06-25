"""Work in progress (not tested)!! This module implements a MultiprocessinglRunner runner to run smrt simulations
in parallel using the Python standard library multiprocessing.

Example::

    # Explicit usage to set runner's parameters:
    runner = MultiprocessinglRunner(progressbar=True)
    res = m.run(sensor, snowpack, runner=runner)
"""

import concurrent.futures

from smrt.core import lib


class MultiprocessingRunner(object):
    """Run the simulations on the local machine on all the cores, using the multiprocessing library for parallelism."""

    def __init__(self, n_jobs=-1, max_numerical_threads=1):
        """Multiprocessing is part of Python standard library.

        Args:
            n_jobs: number of parallel jobs to use. If -1, use all available cores.
            max_numerical_threads: maximum number of threads to be used by numerical libraries. See
                :py:func:`~smrt.core.lib.set_max_numerical_threads` for details.

        """
        self.n_jobs = n_jobs

        if max_numerical_threads > 0:
            # it is recommended to set max_numerical_threads to 1, to disable numerical libraries parallelism.
            lib.set_max_numerical_threads(max_numerical_threads)

    def __call__(self, function, argument_list):
        with concurrent.futures.ProcessPoolExector(self.n_jobs) as executor:
            executor.map(lambda args: function(*args), argument_list)
