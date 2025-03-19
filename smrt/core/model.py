"""A model in SMRT is composed of the electromagnetic scattering theory (:py:mod:`smrt.emmodel`) and
the radiative transfer solver (:py:mod:`smrt.rtsolver`).
The :py:mod:`smrt.emmodel` is responsible for computation of the scattering and absorption coefficients and the phase function of a layer.
It is applied to each layer and it is even possible
to choose different emmodel for each layer (for instance for a complex medium made of different materials: snow, soil, water, atmosphere, ...).
The :py:mod:`smrt.rtsolver` is responsible for propagation of the incident or emitted energy through the layers, up to the surface, and eventually
through the atmosphere.

To build a model, use the :py:meth:`make_model` function with the type of emmodel and type of rtsolver as arguments.
Then call the :py:meth:`Model.run` method of the model instance by specifying the sensor (:py:class:`smrt.core.sensor.Sensor`),
snowpack (:py:class:`smrt.core.snowpack.Snowpack`) and optionally atmosphere (see :py:mod:`smrt.atmosphere`).
The results are returned as a :py:class:`~smrt.core.result.Result` which can then been interrogated to retrieve brightness temperature,
backscattering coefficient, and other information.

Example::

    m = make_model("iba", "dort")

    result = m.run(sensor, snowpack)  # sensor and snowpack are created before

    print(result.TbV())

The model can be run on a list of snowpacks or even more conveniently on a `pandas.Series` or `pandas.DataFrame` including snowpacks.
The first advantage is that by setting parallel_computation=True, the :py:meth:`Model.run` method performs the simulation in parallel
on all the available cores of your machine and even possibly remotely on a high performance cluster using dask.
The second advantage is that the returned :py:class:`~smrt.core.result.Result` object contains all the simulations
and provide an easier way to plot the results or compute statistics.

If a list of snowpacks is provided, it is recommended to also set the snowpack_dimension argument. It takes the form of a tuple
(list of snowpack_dimension values, dimension name). The name and values are used to define the coordinates in the
:py:class:`~smrt.core.result.Result` object. This is useful with timeseries or sensitivity analysis for instance.

Example::

    snowpacks = []
    times = []
    for file in filenames:
        #  create a snowpack for each time series
        sp = ...
        snowpacks.append(sp)
        times.append(sp)

    # now run the model

    res = m.run(sensor, snowpacks, snowpack_dimension=('time', times))

The `res` variable has now a coordinate `time` and res.TbV() returns a timeseries.

Using `pandas.Series` offers an even more elegant way to run SMRT and assemble the results of all the simulations.

    thickness_list = np.arange(0, 10, 1)
    snowpacks = pd.Series([make_snowpack(thickness=t, ........) for t in thickness_list], index=thickness_list)
    # snowpacks is a pandas Series of snowpack objects with the thickness as index

    # now run the model

    res = m.run(sensor, snowpacks, parallel_computation=True)

    # convert the result into a datframe
    res = res.to_dataframe()

The `res` variable is a dataframe with the thickness as index and the channels of the sensor as column.

Using `pandas.DataFrame` is similar. One column must contain Snowpack objects (see snowpack_column argument).
The results of the simulations are automatically joined with this dataframe and returned by
:py:meth:`~smrt.core.result.PassiveResults.to_dataframe` or :py:meth:`~smrt.core.result.ActiveResults.to_dataframe`.

    # df is a DataFrame with several parameters in each row.

    # add a snowpack object for each row
    df['snowpack'] = [make_snowpack(thickness=row['thickness'], ........) for i, row in df.iterrows()]]

    # now run the model
    res = m.run(sensor, snowpacks, parallel_computation=True)

    # convert the result into a datframe
    res = res.to_dataframe()

The `res` variable is a `pandas.DataFrame` equal to df  +  the results at all sensor's channel added.

Most rtsolvers and some emmodels take arguments (usually optional but still useful) that can be specified in two ways in
make_model. Either using the `rtsolver_options` and `emmodel_options` arguments of that function or using the functions
:py:func:`make_rtsolver` and :py:func:`make_emmodel` to build a new class where the prescribed options are applied by default.

Examples of usage::

    make_model("iba", "dort", rtsolver_options=dict(n_max_stream=128))   # original approach to specify options

    make_model("iba", make_rtsolver("dort", n_max_stream=128))                # newer approach that is more readible

Both are equivalent. There is no plan to depreciate the original approach that has some nice use-cases.
"""

from typing import Type, Union
from collections.abc import Sequence, Mapping
import itertools
import inspect

import pandas as pd

from .error import SMRTError
from .result import concat_results
from .plugin import import_class
from .sensor import SensorBase
from .sensitivity_study import SensitivityStudy
from smrt.core import lib


def make_model(
    emmodel, rtsolver=None, emmodel_options=None, rtsolver_options=None, emmodel_kwargs=None, rtsolver_kwargs=None
):
    """create a new model with a given EM model and RT solver. The model is then ready to be run using the :py:meth:`Model.run` method.
    This function is the privileged way to create models compared to class instantiation.
    It supports automatic import of the emmodel and rtsolver modules.

    :param emmodel: type of emmodel to use. Can be given by the name of a file/module in the emmodel directory (as a string) or a class.
        List (and dict, respectively) can be provided when a different emmodel is needed for every layer (or every kind of layer medium).
    :type emmodel:  string or class or list of strings or classes or dict of strings or classes.
        If a list of emmodels is given, the size must be the same as the number of layers in the snowpack.
        If a dict is given, the keys are the kinds of medium and the values are the associated emmodels to each sort of medium.
        The layer attribute 'medium' is used to determine the emmodel to use for each layer.
    :type emmodel:  string or class; or list of strings or classes; or dict of strings or classes.
    :param rtsolver: type of RT solver to use. Can be given by the name of a file/module in the rtsolver directeory (as a string)
        or a class. This argument is optional when only the computation of the layer electromagnetic properties is needed.
    :type rtsolver: string or class.
    :param emmodel_options: arguments used to create the emmodel instance of each layer. Valid arguments depend on the
        selected emmodel (refer to the documentation of the selected emmodel).
        The function :py:func:`emmodel` provides an alternative to setting `emmodel_options`.
    :type emmodel_options: dict or a list of dict. In the latter case, the size of the list must be the same as
        the number of layers in the snowpack.
    :param rtsolver_options: arguments used to create the rtsolver instance (refer to the documentation of the rtsolvers). The function
        :py:func:`rtsolver` provides an alternative to setting `rtsolver_options`.
    :type rtsolver_options: dict

    :returns: a model instance
    """

    if emmodel_kwargs is not None:
        raise DeprecationWarning('Use emmodel_options instead of emmodel_kwargs')
        emmodel_options = emmodel_kwargs

    if rtsolver_kwargs is not None:
        raise DeprecationWarning('Use rtsolver_options instead of rtsolver_kwargs')
        rtsolver_options = rtsolver_kwargs

    return Model(emmodel, rtsolver, emmodel_options=emmodel_options, rtsolver_options=rtsolver_options)


def make_rtsolver(rtsolver_class: Union[str, Type], **options) -> Type:
    """return a rtsolver subclass of cls (either given as a string or a class) where the provided options are applied to __init__.
    This function provides an alternative to setting `rtsolver_options` in :py:func:`make_model`).

    Example::
        make_model(..., make_rtsolver("dort", n_max_stream=128))
    """
    return lib.class_specializer('rtsolver', rtsolver_class, **options)


def make_emmodel(emmodel_class: Union[str, Type], **options) -> Type:
    """return a emmodel subclass of cls (either given as a string or a class) where the provided options are applied to __init__.
    This function provides an alternative to setting `emmodel_options` in :py:func:`make_model`).

    Example::
        make_model(make_emmodel("iba", dense_snow_correction=True), ...)
    """
    return lib.class_specializer('emmodel', emmodel_class, **options)


def get_emmodel(emmodel):
    """return an emmodel class from the file name 'emmodel'"""

    raise DeprecationWarning("This function will be remove soon, use make_emmodel instead.")
    if isinstance(emmodel, str):
        emmodel = import_class('emmodel', emmodel)
    assert inspect.isclass(emmodel)
    return emmodel


def make_emmodel_instance(emmodel, sensor, layer, **emmodel_options):
    """create a new emmodel instance based on the emmodel class or string. This function used to be called `make_emmodel`
    but has been renamed from SMRT v1.4 and will soon be depreciated. It is recommended to use instead::

        em = make_emmodel(emmodel)(sensor, layer, **emmodel_options)

    or::

        emmodel_class = make_emmodel(emmodel)
        em = emodel_class(sensor, layer, **emmodel_options)

    :param emmodel: type of emmodel to use. Can be given by the name of a file/module in the emmodel directory (as a string) or a class.
    :param sensor: sensor to use for the calculation.
    :param layer: layer to use for the calculation
    """

    # instantiate
    emmodel = make_emmodel(emmodel)  # get the class
    if not isinstance(sensor, SensorBase):
        raise SMRTError("the first argument of 'run' must be a sensor")
    return emmodel(sensor, layer, **emmodel_options)  # create a emmodel


class Model(object):
    """This class drives the whole calculation"""

    def __init__(self, emmodel, rtsolver, emmodel_options=None, rtsolver_options=None):
        """create a new model. It is not recommended to instantiate Model class directly. Instead use the :py:meth:`make_model` function."""

        # emmodel can be a single value (class or string), an array with the same size as snowpack layers array, or a
        # mapping between an emmodel for each layer medium

        super().__init__()

        if lib.is_sequence(emmodel):
            self.emmodel = [make_emmodel(em) for em in emmodel]
        elif isinstance(emmodel, Mapping):
            self.emmodel = {k: make_emmodel(em) for k, em in emmodel.items()}
        else:
            self.emmodel = make_emmodel(emmodel)

        if isinstance(rtsolver, str):
            self.rtsolver = import_class('rtsolver', rtsolver)
        else:
            self.rtsolver = rtsolver

        # The implementation avoid metaclass by supplying an optional list of arguments to the emmodel and rtsolver
        # to alter the behavior the emmodel (or rtsolver)
        # this is not the most general case, but metaclass can still be used for advanced user

        self.emmodel_options = emmodel_options if emmodel_options is not None else dict()
        self.rtsolver_options = rtsolver_options if rtsolver_options is not None else dict()

    def set_rtsolver_options(self, options=None, **kwargs):
        """set the option for the rtsolver"""
        if options is not None:
            if not isinstance(options, Mapping):
                raise SMRTError('options must be a Mapping (eg. dict)')
            self.rtsolver_options = dict(options)  # overload the options

        self.rtsolver_options.update(kwargs)  # update the options

    def set_emmodel_options(self, options=None, **kwargs):
        """set the options for the emmodel"""
        if options is not None:
            if not isinstance(options, Mapping):
                raise SMRTError('options must be a Mapping (eg. dict)')
            self.emmodel_options = dict(options)  # overload the options

        self.emmodel_options.update(kwargs)  # update the options

    def run(
        self,
        sensor,
        snowpack,
        atmosphere=None,
        snowpack_dimension=None,
        snowpack_column='snowpack',
        progressbar=False,
        parallel_computation=False,
        runner=None,
    ):
        """Run the model for the given sensor configuration and return the results

        :param sensor: sensor to use for the calculation. Can be a list of the same size as the snowpack list.
            In this case, the computation is performed for each pair (sensor, snowpack).
        :param snowpack: snowpack to use for the calculation. Can be a single snowpack, a list of snowpack, a dict of snowpack or
            a SensitivityStudy object.
        :param snowpack_dimension: name and values (as a tuple) of the dimension to create for the results when a list of snowpack
            is provided. E.g. time, point, longitude, latitude. By default the dimension is called 'snowpack' and the values are
            rom 1 to the number of snowpacks.
        :param snowpack_column: when snowpack is a DataFrame this argument is used to specify which column contians the Snowpack objects
        :param progressbar: if True, display a progress bar during multi-snowpacks computation
        :param parallel_computation: if True, use the joblib library to run the simulations of many snowpacks in parallel.
            Otherwise, the simulations are run sequentially, one after one. See 'runner' for a more advanced control
            on parallel computations. Note for users seeking performances: numpy and scipy usually also perform low-
            level parallel computations
            that may (inefficiently) interact with the high-level parallelism activated by parallel_computation. For this reason
            joblib and other parallel runners try to desactivate numpy and scipy low-level parallelism (see
            :py:func:`~smrt.core.lib.set_max_numerical_threads`) to maximize performances. Conversely it means that
            when parallel_computation is False, the simulations are run sequentially, but numpy and scipy
            parallelism is NOT disabled. If you really want to use a single core for the simulations, you must first call
            :py:func:`~smrt.core.lib.set_max_numerical_threads` with 1 as argument and then call Model.run with
            parallel_computation=False.
        :param runner: a 'runner' is a function (or more likely a class with a __call__ method) that takes a function and a
            list/generator of simulations, executes the function on each simulation and returns a list of results.
            'parallel_computation' allows to select between two default (basic) runners (sequential and joblib).
            Use 'runner' for more advanced parallel distributed computations. To develop a costum runner, see the implementation of
            :py:class:`JoblibParallelRunner` for instance.
        :returns: result of the calculation(s) as a :py:class:`Results` instance
        """

        if atmosphere is not None:
            raise DeprecationWarning("""The atmosphere argument of the run method is depreciated.
Setting the 'atmosphere' through make_snowpack (and similar functions) or using medium = atmosphere + snowpack are now the recommended ways.""")

        if not (
            isinstance(sensor, SensorBase)
            or (lib.is_sequence(sensor) and all(isinstance(s, SensorBase) for s in sensor))
        ):
            raise SMRTError("the first argument of 'run' must be a sensor or a sequence of sensor")

        # determine the simulations to run
        simulations, dimensions = self.prepare_simulations(sensor, snowpack, snowpack_dimension, snowpack_column)

        # determine the runner
        if runner is None:
            if parallel_computation:
                runner = JoblibParallelRunner(progressbar=progressbar)
            else:
                runner = SequentialRunner(progressbar=progressbar)

        #  run all the simulations (with atmosphere as long as it is not depreciated), the results is a flat list of results
        results = runner(self.run_single_simulation, ((simul, atmosphere) for simul in simulations))
        results = list(results)  # consume the generator if it is one

        # reshape the results with successive concatenations
        for dimension in reversed(dimensions):
            n = len(dimension[1]) if isinstance(dimension, tuple) else len(dimension)
            assert n > 0, f'dimension={dimensions}'
            results = [concat_results(results[i : i + n], dimension) for i in range(0, len(results), n)]

        assert len(results) == 1
        results = results[0]

        if isinstance(snowpack, pd.DataFrame):
            # remove the snowpack_column
            results.mother_df = snowpack.drop(snowpack_column, axis=1)

        return results

    def prepare_simulations(self, sensor, snowpack, snowpack_dimension, snowpack_column):
        # return a flat list of pairs (sensor, snowpack). Each is a unique simulation. The second returned parameter
        # is the list of (axis, values) to be used to concatenate the results.

        # determine if we have several snowpacks
        # is it a SensitivityStudy object ?
        if isinstance(snowpack, SensitivityStudy):
            snowpack_dimension = snowpack.variable, snowpack.values
            snowpack = snowpack.snowpacks.tolist()

        # or is it a dict ?
        if isinstance(snowpack, Mapping):
            snowpack_dimension = 'snowpack', list(snowpack.keys())
            snowpack = list(snowpack.values())

        if isinstance(snowpack, pd.DataFrame):
            try:
                snowpack = snowpack[snowpack_column]
            except KeyError:
                raise SMRTError(
                    "the snowpack DataFrame has no column named '%s'. Check the snowpack_column argument."
                    % snowpack_column
                )
            assert isinstance(snowpack, pd.Series)

        # or is it a pandas Series ?
        if isinstance(snowpack, pd.Series):
            name = snowpack.index.name
            if name is None:
                name = 'snowpack'
            snowpack_dimension = name, snowpack.index.tolist()
            snowpack = snowpack.tolist()

        # or a sequence ?
        if lib.is_sequence(snowpack):
            if snowpack_dimension is None:
                snowpack_dimension = 'snowpack', None
            if snowpack_dimension[1] is None:
                snowpack_dimension = snowpack_dimension[0], range(len(snowpack))

        if (
            (snowpack_dimension is not None)
            and isinstance(snowpack, tuple)
            and (len(snowpack) != len(snowpack_dimension[1]))
        ):
            raise SMRTError('The list of snowpacks must have the same length as the snowpack_dimension')

        if isinstance(snowpack_dimension, tuple) and not isinstance(snowpack_dimension[0], str):
            raise SMRTError("When the 'snowpack_dimension' argument is a tuple, the first argument must be a string")

        # the sensor object is split in its basic sensors (config). How deep the sensor is split depends on the
        # radiative transfer solver's broadcast capability.

        def get_sensor_configurations(sensor):
            rt_solver_broadcast_capability = getattr(self.rtsolver, '_broadcast_capability', [])
            sensor_configurations = [
                (axis, values)
                for (axis, values) in sensor.configurations()
                if axis not in rt_solver_broadcast_capability
            ]
            return sensor_configurations

        def prepare_recursive(sensor, sensor_configurations, snowpack):
            """return the cross product of sensor x snowpack"""
            if sensor_configurations:
                axis, values = sensor_configurations[0]
                for sensor_subset in sensor.iterate(axis):
                    yield from prepare_recursive(sensor_subset, sensor_configurations[1:], snowpack)
            else:  # we're at the end
                if lib.is_sequence(snowpack):
                    for sp in snowpack:
                        yield (sensor, sp)
                else:
                    yield (sensor, snowpack)

        if lib.is_sequence(sensor):
            # sensor is a list
            if len(sensor) != len(snowpack):
                raise SMRTError('when sensor is a sequence, the length must be the same as snowpack sequence length')
            sensor_configurations = get_sensor_configurations(
                next(iter(sensor))
            )  # take the config of the first, assume all are the same.
            # we should check that all the configurations are the same...
            simulations = (prepare_recursive(se, sensor_configurations, sp) for se, sp in zip(sensor, snowpack))
            simulations = list(itertools.chain(*simulations))  # flatten
        else:
            # normal case
            sensor_configurations = get_sensor_configurations(sensor)
            simulations = prepare_recursive(
                sensor, sensor_configurations.copy(), snowpack
            )  # I don't know why I put a .copy here...

        dimensions = sensor_configurations
        if snowpack_dimension is not None:
            dimensions.append(snowpack_dimension)

        return simulations, dimensions

    def prepare_emmodels(self, sensor, snowpack):
        # return emmodels instances for each layer

        if lib.is_sequence(self.emmodel):
            # check we have the same number as layer in the snowpack
            assert len(self.emmodel) == snowpack.nlayer  # check we have the same number as layer in the snowpack
            # one different model per layer
            emmodel_list = self.emmodel
        elif isinstance(self.emmodel, Mapping):
            emmodel_list = (self.emmodel[layer.medium] for layer in snowpack.layers)
        else:
            # the same model for all layers
            emmodel_list = itertools.cycle([self.emmodel])

        if isinstance(self.emmodel_options, Sequence):
            assert (
                len(self.emmodel_options) == snowpack.nlayer
            )  # check we have the same number as layer in the snowpack
            emmodel_options_list = self.emmodel_options
        elif (
            isinstance(self.emmodel, Mapping)
            and (self.emmodel_options)
            and all(isinstance(options, Mapping) for options in self.emmodel_options.values())
        ):
            emmodel_options_list = (self.emmodel_options[layer.medium] for layer in snowpack.layers)
        else:
            emmodel_options_list = itertools.cycle([self.emmodel_options])

        # create a list of emmodel instances (ready to run)
        emmodel_instances = [
            make_emmodel_instance(emmodel, sensor, layer, **emmodel_options)
            for emmodel, emmodel_options, layer in zip(emmodel_list, emmodel_options_list, snowpack.layers)
        ]

        return emmodel_instances

    def run_single_simulation(self, simulation, atmosphere):
        # run a single simulation
        sensor, snowpack = simulation

        emmodel_instances = self.prepare_emmodels(sensor, snowpack)

        if self.rtsolver is not None:
            # need to create the rtsolver ?
            if inspect.isclass(self.rtsolver):
                rtsolver = self.rtsolver(**self.rtsolver_options)  # create with arguments
            else:
                if not getattr(self.rtsolver, '_reentrant', False):
                    raise SMRTError('This solver can not be used with an instance')
                # no use the instance as it is.
                # this instances has possible memory of the last solve... and this is INCOMPATIBLE with // computation for most solver)
                # In the future this feature should be either removed or at least restricted when the // computation will be activate.
                rtsolver = self.rtsolver

            # run the rtsolver
            result = rtsolver.solve(snowpack, emmodel_instances, sensor, snowpack.atmosphere or atmosphere)

            return result

    def run_later(self, sensor, snowpack, **kwargs):
        from .run_promise import RunPromise  # local import to avoid start time

        return RunPromise(self, sensor, snowpack, kwargs)


class SequentialRunner(object):
    """Run the simulations sequentially on a single (local) core. This is the most simple way to run smrt simulations, but the
    efficiency is poor."""

    def __init__(self, progressbar):
        """
        :param progressbar: show a progress bar if True
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
                    desc='Running SMRT',
                )
            ]
        else:
            return [function(*args) for args in argument_list]


class JoblibParallelRunner(object):
    """Run the simulations on the local machine on all the cores, using the joblib library for parallelism."""

    def __init__(self, progressbar, backend='loky', n_jobs=-1, max_numerical_threads=1):
        """Joblib is a lightweight library for embarasingly parallel task.

        :param progressbar: show a progress bar if True
        :param backend: see joblib documentation. The default 'loky' is the recommended backend.
        :param n_jobs: see joblib documentation. The default is to use all the cores.
        :param max_numerical_threads: :py:func:`~smrt.core.lib.set_max_numerical_threads`. The default avoid miximing different
        parallelism techniques.

        """
        self.n_jobs = n_jobs
        self.backend = backend
        self.progressbar = progressbar

        if max_numerical_threads > 0:
            # it is recommended to set max_numerical_threads to 1, to disable numerical libraries parallelism.
            lib.set_max_numerical_threads(max_numerical_threads)

    def __call__(self, function, argument_list):
        from joblib import Parallel, delayed

        if self.progressbar:
            from tqdm.auto import tqdm

            runner = Parallel(return_as='generator', n_jobs=self.n_jobs, backend=self.backend)  # Parallel Runner

            argument_list = list(argument_list)
            return list(
                tqdm(
                    runner(delayed(function)(*args) for args in argument_list),
                    total=len(argument_list),
                    desc='Running SMRT in parallel',
                )
            )
        else:
            runner = Parallel(n_jobs=self.n_jobs, backend=self.backend)  # Parallel Runner
            return runner(delayed(function)(*args) for args in argument_list)
