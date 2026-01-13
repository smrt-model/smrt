"""This module implements a joblib runner tp execute smrt simulations in parallel. It is used by default if parallel_computation=True
in py:func:`~smrt.core.model.make_model` function.
"""


class JoblibParallelRunner(object):
    """
    Run the simulations on the local machine on all the cores, using the joblib library for parallelism.
    """

    def __init__(self, progressbar, backend="loky", n_jobs=None, max_numerical_threads=1):
        """
        Build a joblib parallel runner.

        Joblib is a lightweight library for embarasingly parallel task.

        Args:
        progressbar: show a progress bar if True
            backend: see joblib documentation. The default 'loky' is the recommended backend.
            n_jobs (int): see joblib documentation. The default is to use all the cores.
            max_numerical_threads: :py:func:`~smrt.core.lib.set_max_numerical_threads`. The default avoid mixing different
                parallelism techniques.

        """
        from joblib import cpu_count  # should become a lazy import in the future

        if n_jobs is None:
            n_jobs = cpu_count(only_physical_cores=False) // max_numerical_threads

        self.n_jobs = n_jobs
        self.backend = backend
        self.progressbar = progressbar

        # the following is done internally by joblib...
        # if max_numerical_threads > 0:
        #     # it is recommended to set max_numerical_threads to 1, to disable numerical libraries parallelism.
        #     lib.set_max_numerical_threads(max_numerical_threads)

    def __call__(self, function, argument_list):
        """
        Run the function on all the argument_list in parallel.

        Args:
            function: function to run on each argument
            argument_list: list of arguments to pass to the function

        Returns:
            list: list of results from the function
        """

        from joblib import Parallel, delayed  # should become a lazy import in the future

        if self.progressbar:
            from tqdm.auto import tqdm

            runner = Parallel(return_as="generator", n_jobs=self.n_jobs, backend=self.backend)  # Parallel Runner

            argument_list = list(argument_list)
            return list(
                tqdm(
                    runner(delayed(function)(*args) for args in argument_list),
                    total=len(argument_list),
                    desc="Running SMRT in parallel",
                )
            )
        else:
            runner = Parallel(n_jobs=self.n_jobs, backend=self.backend)  # Parallel Runner
            return runner(delayed(function)(*args) for args in argument_list)
