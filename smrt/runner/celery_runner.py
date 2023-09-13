"""Run the simulations using Celery on a cluster. This requires  setup on the cluster
(see the Celery documentation https://docs.celeryproject.org/).

Example::

    from smrt.runner.celery import CeleryParallelRunner

    runner = CeleryParallelRunner()   # run with the default broker redis://localhost:6379/0 but any url can be provided, as well as a Celery object

    m = make_model(...)

    m.run(sensor, snowpack, runner=runner)

"""


from smrt.core import lib
from celery import shared_task, Celery, group

app = Celery('smrt_runner', broker="redis://localhost:6379")

app.conf.update(
    result_backend="redis",
    result_serializer="pickle",
    result_expires=3600,
    accept_content=['pickle']
)


class CeleryParallelRunner(object):
    """Run the simulations using dask.distributed on a cluster.
    """

    def __init__(self, broker="redis://localhost:6379", chunk=10):
        """prepare a dask runner.

        :param broker: the url or a Celery object (app)
        :param chunk: size of the chunk to transmit to the runner

        """

        super().__init__()

        # if isinstance(broker, str):
        #    self.app = Celery('hello', broker=broker)
        # else:
        #    self.app = broker

        self.chunk = chunk

    def __call__(self, function, argument_list):

        # make a bag
        argument_list = list(argument_list)

        # chunck does not work with pickle serializer. It seems the option of the function are not transmitted to the chunk
        # iterator = [(function, *arg) for arg in argument_list]
        # task_group = _celery_call_with_single_numerical_threads.chunks(iterator, ntask) #.group()

        # chunk ourselves:
        tasks = [_celery_call_with_single_numerical_threads.s(function, argument_list[i: i + self.chunk])
                 for i in range(0, len(argument_list), self.chunk)]
        tasks = group(tasks)

        # without chunk:
        # tasks = group([_celery_call_with_single_numerical_threads.s(function, *args) for args in argument_list])

        results = tasks.apply_async().get()

        # flatten the results
        results = [item for sublist in results for item in sublist]

        assert len(results) == len(argument_list)

        return results


@shared_task(name='celery.run_smrt', serializer='pickle', result_serializer='pickle', acks_late=True)
def _celery_call_with_single_numerical_threads(func, argument_list):
    lib.set_max_numerical_threads(1)
    return [func(*args) for args in argument_list]
