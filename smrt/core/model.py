
"""A model in SMRT is composed of the electromagnetic scattering theory (:py:mod:`smrt.emmodel`) and 
the radiative transfer solver (:py:mod:`smrt.rtsolver`).
The :py:mod:`smrt.emmodel` is responsible for computation of the scattering and absorption coefficients and the phase function of a layer.
It is applied to each layer and it is even possible
to choose different emmodel for each layer (for instance for a complex medium made of different materials: snow, soil, water, atmosphere, ...).
The :py:mod:`smrt.rtsolver` is responsible for propagation of the incident or emitted energy through the layers, up to the surface, and eventually 
through the atmosphere.

To build a model, use the :py:func:`make_model` function with the type of emmodel and type of rtsolver as arguments.
Then call the :py:meth:`Model.run` method of the model instance by specifying the sensor (:py:class:`smrt.core.sensor.Sensor`),
snowpack (:py:class:`smrt.core.snowpack.Snowpack`) and optionally atmosphere (see :py:mod:`smrt.atmosphere`).
The results are returned as a :py:class:`~smrt.core.result.Result` which can then been interrogated to retrieve brightness temperature,
backscattering coefficient, etc.

Example::

    m = make_model("iba", "rtsolver")

    result = m.run(sensor, snowpack)  # sensor and snowpack are created before

    print(result.TbV())

The :py:meth:`~Model.run` method can be used with list of snowpacks. In this case, it is recommended to set the snowpack_dimension_name and 
snowpack_dimension_values variable which gives the name and values of the coordinates that are create for the Results. This is useful with
timeseries for instance.

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

"""

import collections
import inspect
import copy
import six

import numpy as np

from .error import SMRTError
from .result import concat_results
from .plugin import import_class
from .sensitivity_study import SensitivityStudy



def make_model(emmodel, rtsolver, emmodel_kwargs=None, rtsolver_kwargs=None):
    """create a new model with a given EM model and RT solver. The model is then ready to be run using the :py:meth:`Model.run` method. This function is the privileged way
    to create models compared to class instantiation. It supports automatic import of the emmodel and rtsolver modules.

    :param emmodel: type of emmodel to use. Can be given by the name of a file/module in the emmodel directory (as a string) or a class.
    :type emmodel:  string or class or list of strings or classes. If a list is given, different models are used for the different layers of the snowpack. In this case, the size of the list must be the same as the number of layers in the snowpack.
    :param rtsolver: type of solver to use. Can be given by the name of a file/module in the rtsolver directeory (as a string) or a class.
    :type rtsolver: string or class
    :param emmodel_kwargs: extra arguments to use to create emmodel instance. Valid arguments depend on the selected emmodel. It is documented in for each emmodel class.
    :type emmodel_kwargs: dict or a list of dict. In the latter case, the size of the list must be the same as the number of layers in the snowpack.
    :param rtsolver_kwargs: extra to use to create the rtsolver instance (see __init__ of the solver used).
    :type rtsolver_args: dict

    :returns: a model instance
    """

    return Model(emmodel, rtsolver, emmodel_kwargs=emmodel_kwargs, rtsolver_kwargs=rtsolver_kwargs)


class Model(object):
    """ This class drives the whole calculation
    """
    def __init__(self, emmodel, rtsolver, emmodel_kwargs=None, rtsolver_kwargs=None):
        """create a new model. It is not recommanded to instantiate Model class directly. Instead use the :py:meth:`make_model` function.
        """

        if isinstance(emmodel, six.string_types):
            self.emmodel = import_class(emmodel, root='smrt.emmodel')
        else:
            self.emmodel = emmodel  # can be a single value or an array with the same size as snowpack layers array

        if isinstance(rtsolver, six.string_types):
            self.rtsolver = import_class(rtsolver, root='smrt.rtsolver')
        else:
            self.rtsolver = rtsolver

        # The implementation avoid metaclass by supplying an optional list of arguments to the emmodel and rtsolver
        # to alter the behavior the emmodel (or rtsolver)
        # this is not the most general case, but metaclass can still be used for advanced user
        self.emmodel_kwargs = emmodel_kwargs if emmodel_kwargs is not None else dict()
        self.rtsolver_kwargs = rtsolver_kwargs if rtsolver_kwargs is not None else dict()

    def run(self, sensor, snowpack, atmosphere=None, snowpack_dimension=None):
        """ Run the model for the given sensor configuration and return the results

            :param sensor: sensor to use for the calculation
            :param snowpack: snowpack to use for the calculation. Can be a singel snowpack, a list or a SensitivityStudy object.
            :param snowpack_dimension: name and values (as a tuple) of the dimension to create for the results when a list of snowpack is provided. E.g. time, point, longitude, latitude. By default the dimension is called 'snowpack' and the values are from 1 to the number of snowpacks.
            :returns: result of the calculation(s) as a :py:class:`Results` instance
        """

        # first determine which dimension we must iterate on in this routine
        for dim in ["frequency", "theta_inc", "polarization_inc", "theta", "phi", "polarization"]:
            # do we need to iterate on this dimension ?
            values = getattr(sensor, dim)
            if (isinstance(values, np.ndarray) or isinstance(values, collections.Sequence)) and \
                (not hasattr(self.rtsolver, "_broadcast_capability") or dim not in self.rtsolver._broadcast_capability):

                result_list = []
                for x in values:  # iterate over the values
                    sensor_subset = copy.copy(sensor)  # shallow copy... hope sensor attributes are immutable!!
                    setattr(sensor_subset, dim, x)  # change the sensor
                    res = self.run(sensor_subset, snowpack, atmosphere=atmosphere, snowpack_dimension=snowpack_dimension)  # recursive call
                    result_list.append(res)

                return concat_results(result_list, (dim, values))

        # second determine if we have several snowpack
        if isinstance(snowpack, SensitivityStudy):
            snowpack_dimension = (snowpack.variable, snowpack.values)
            snowpack = snowpack.snowpacks.tolist()

        if isinstance(snowpack, collections.Sequence):
            if snowpack_dimension is None:
                dimension_name, dimension_values = "Snowpack", None
            else:
                dimension_name, dimension_values = snowpack_dimension
            if dimension_values is None:
                dimension_values = range(len(snowpack))

            result_list = [self.run(sensor, sp, atmosphere=atmosphere) for sp in snowpack]  # parallel computation would be better !
            return concat_results(result_list, (dimension_name, dimension_values))


        # not need to iterate anymore, either because the solver deals with the dimension or sensor config has single values.
        # prepare to run

        # create a list of emmodel instances (ready to run)
        emmodel_instances = list()
        if isinstance(self.emmodel, collections.Sequence):  # got a list of emmodel. TODO: adapt to iterator.
            # check we have the same number as layer in the snowpack
            assert(len(self.emmodel) == snowpack.nlayer)

            # check these are different instances or instantiate them
            for i, emmodel, lay in enumerate(zip(self.emmodel, snowpack.layers)):
                assert inspect.isclass(emmodel)
                if isinstance(self.emmodel_kwargs, collections.Sequence):
                    emmodel_kwargs = self.emmodel_kwargs[i]
                else:
                    emmodel_kwargs = self.emmodel_kwargs
                # instantiate
                emmodel_instances.append(emmodel(sensor, snowpack.layers, **emmodel_kwargs))  # create a emmodele
        else:  # the same model for all the layers
            for lay in snowpack.layers:
                emmodel_instances.append(self.emmodel(sensor, lay, **self.emmodel_kwargs))

        # need to create the rtsolver ?
        if inspect.isclass(self.rtsolver):
            rtsolver = self.rtsolver(**self.rtsolver_kwargs)  # create with arguments
        else:
            # no use the instance as it is (with possible memory of the last solve...)
            rtsolver = self.rtsolver

        # run the rtsolver
        result = rtsolver.solve(snowpack, emmodel_instances, sensor, atmosphere)

        return result
