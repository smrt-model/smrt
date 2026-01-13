"""This module implements a sequential runner for smrt simulations. It is used by default if parallel_computation=False
in py:func:`~smrt.core.model.make_model` function."""


class SequentialRunner(object):
    """
    Run the simulations sequentially on a single (local) core. This is the most simple way to run smrt simulations, but
    the efficiency is poor.

    """

    def __init__(self, progressbar, max_numerical_threads=1):
        """
        Build a sequential runner

        Args:
          progressbar: show a progress bar if True.
          max_numerical_threads: see :py:func:`~smrt.core.lib.set_max_numerical_threads`. The default avoid mixing different

        """
        self.progressbar = progressbar

    def __call__(self, function, argument_list):
        if self.progressbar:
            from tqdm.auto import tqdm

            argument_list = list(argument_list)
            return [
                function(*args)
                for args in tqdm(
                    argument_list,
                    total=len(argument_list),
                    desc="Running SMRT",
                )
            ]
        else:
            return [function(*args) for args in argument_list]
