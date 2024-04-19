"""Run the simulations using dask.distributed on a cluster. This requires  setup on the cluster
(see the dask.distributed documentation).

Example::

    from smrt.runner.dask import DaskParallelRunner

    runner = DaskParallelRunner()   # run on localhost:7454 by default but an url can be provided

    m = make_model(...)

    m.run(sensor, snowpack, runner=runner)

"""


from smrt.core import lib
from dask.distributed import Client


class DaskParallelRunner(object):
    """Run the simulations using dask.distributed on a cluster.
    """

    def __init__(self, progressbar=False, client="localhost:7454", chunk=10):
        """prepare a dask runner.

        :param progressbar: show a progress bar if True (not available for DaskparalleRunner)
        :param client: the url or a dask client objbect
        :param chunk: size of the chunk to transmit to the runner

        """

        super().__init__()

        if isinstance(client, str):
            self.client = Client(client, set_as_default=False)
        else:
            self.client = client

        self.chunk = chunk

    def __call__(self, function, argument_list):

        def function_with_single_numerical_threads(args):
            lib.set_max_numerical_threads(1)
            return function(*args)

        # make a bag
        argument_list = list(argument_list)

        # prepare the futures with chunks
        futures = [self.client.map(function_with_single_numerical_threads, argument_list[i: i + self.chunk])
                   for i in range(0, len(argument_list), self.chunk)]

        results = self.client.gather(futures, direct=False)

        # flatten the list of results if necessary
        ret = []
        for res in results:
            if isinstance(res, list):
                ret += res
            else:
                ret.append(res)

        return ret
