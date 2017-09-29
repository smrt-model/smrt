# coding: utf-8

""" SensitivityStudy is used to easily conduct sensitivity studies.

Example::

    times = [datetime(2012, 1, 1), datetime(2012, 1, 5), , datetime(2012, 1, 10)]
    snowpacks = SensitivityStudy("time", times, [snowpack_1jan, snowpack_5jan, snowpack_10jan])


    res = model.run(sensor, snowpacks)

The `res` variable is a :py:class:`Result` instance, so that for all the methods of this class that can be called, they will return a timeseries.
For instance result.TbV(theta=53) returns a time-series of brightness temperature at V polarization and 53° incidence angle and the following code
plots this timeseries::

    plot(times, result.TbV(theta=53))

"""

import numpy as np
import xarray as xr


class SensitivityStudy(object):

    def __init__(self, name, values, snowpacks):
        #super(self.__class__, self).__init__(snowpacks, dims=[name], coords={name: values})

        self.snowpacks = np.array(snowpacks)
        self.variable = name
        self.values = np.array(values)

    def __getitem__(self, key):
        return type(self)(self.variable, self.values[key], self.snowpacks[key])


def sensitivity_study(name, values, snowpacks):
    """create a sensitivity study

    :param name: name of the variable to investigate
    :param values: values taken by the variable
    :param snowpacks: list of snowpacks. Can be a sequence or a function that takes one argument and return a snowpack.
    In the latter case, the function is called for each values to build the list of snowpacks"""

    if callable(snowpacks):
        snowpacks = [snowpacks(value) for value in values]

    return SensitivityStudy(name, values, snowpacks)
