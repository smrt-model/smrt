# coding: utf-8

""" The results of RT Solver are hold by the :py:class:`Result` class. This class provides several functions
to access to the Stokes Vector and Muller matrix in a simple way. Most notable ones are :py:meth:`Result.TbV` and :py:meth:`Result.TbH` for the passive mode calculations and
:py:meth:`Result.sigmaHH` and :py:meth:`Result.sigmaVV`. Other methods could be developed for specific needs.

To save results of calculations in a file, simply use the pickle module or other serialization schemes. We may provide a unified and inter-operable solution in the future.

Under the hood, :py:class:`Result` uses xarray module which provides multi-dimensional array with explicit, named, dimensions. Here the common dimensions
are frequency, polarization, polarization_inc, theta_inc, theta, and phi. They are created by the RT Solver. The interest of using named dimension is that slice of the xarray (i.e. results)
can be selected based on the dimension name whereas with numpy the order of the dimensions matters. Because this is very convenient, users may be
interested in adding other dimensions specific to their context such as time, longitude, latitude, points, ... To do so, :py:meth:`smrt.core.model.Model.run` accepts a list of snowpack
and optionally the parameter `snowpack_dimension` is used to specify the name and values of the new dimension to build.

Example::

    times = [datetime(2012, 1, 1), datetime(2012, 1, 5), , datetime(2012, 1, 10)]
    snowpacks = [snowpack_1jan, snowpack_5jan, snowpack_10jan]

    res = model.run(sensor, snowpacks, snowpack_dimension=('time', times))

The `res` variable is a :py:class:`Result` instance, so that for all the methods of this class that can be called, they will return a timeseries.
For instance result.TbV(theta=53) returns a time-series of brightness temperature at V polarization and 53° incidence angle and the following code
plots this timeseries::

    plot(times, result.TbV(theta=53))

"""

# Stdlib import

import numpy as np
import xarray as xr
import pandas as pd


def open_result(filename):
    """read a result save to disk. See :py:meth:`Result.save` method."""
    data = xr.open_dataarray(filename, autoclose=True)

    #  argh... need to convert polarization in unicode!
    for d in data.dims:
        if d.startswith("polarization"):
            data[d] = data[d].astype("U1")

    return Result(data)


class Result(object):
    """ Contains the results of a/many computations and provides convenience functions to access these results

    """

    def __init__(self, intensity, coords=None):
        """Construct results array with the given intensity array (numpy array or xarray) and dimensions if numpy array is given

"""
        if isinstance(intensity, xr.DataArray):
            self.data = intensity
        else:
            self.data = xr.DataArray(intensity, coords)

    @property
    def coords(self):
        return self.data.coords

    def Tb(self, **kwargs):
        """Return brightness temperature. Any parameter can be added to slice the results (e.g. frequency=37e9 or polarization='V'). See xarray slicing with sel method (to document)"""
        return _strongsqueeze(self.data.sel(**kwargs))

    def Tb_as_dataframe(self, **kwargs):
        """Return brightness temperature. Any parameter can be added to slice the results (e.g. frequency=37e9 or polarization='V'). See xarray slicing with sel method (to document)"""
        return self.Tb(**kwargs).to_dataframe(name='Tb')

    def TbV(self, **kwargs):
        """Return V polarization. Any parameter can be added to slice the results (e.g. frequency=37e9). See xarray slicing with sel method (to document)"""
        return _strongsqueeze(self.data.sel(polarization='V', **kwargs))

    def TbH(self, **kwargs):
        """Return H polarization. Any parameter can be added to slice the results (e.g. frequency=37e9). See xarray slicing with sel method (to document)"""
        return _strongsqueeze(self.data.sel(polarization='H', **kwargs))

    def polarization_ratio(self, ratio="H_V", **kwargs):
        """Return polarization ratio. Any parameter can be added to slice the results (e.g. frequency=37e9). See xarray slicing with sel method (to document)"""
        return _strongsqueeze(self.data.sel(polarization=ratio[0], **kwargs) / self.data.sel(polarization=ratio[-1], **kwargs))

    def sigma(self, **kwargs):
        """Return backscattering coefficient. Any parameter can be added to slice the results (e.g. frequency=37e9, polarization_inc='V', polarization='V'). See xarray slicing with sel method (to document)"""
        theta = np.array(self.data.theta)
        return 4*np.pi*np.cos(np.radians(theta))*_strongsqueeze(self.data.sel(**kwargs).sel_points(theta_inc=theta, theta=theta))

    def sigma_as_dataframe(self, **kwargs):
        """Return backscattering coefficient. Any parameter can be added to slice the results (e.g. frequency=37e9, polarization_inc='V', polarization='V'). See xarray slicing with sel method (to document)"""
        return self.sigma(**kwargs).to_dataframe(name='sigma')

    def sigmaVV(self, **kwargs):
        """Return VV backscattering coefficient. any parameter can be added to slice the results (e.g. frequency=37e9). See xarray slicing with sel method (to document)"""
        theta = np.array(self.data.theta)
        return 4*np.pi*np.cos(np.radians(theta))*_strongsqueeze(self.data.sel(polarization_inc='V', polarization='V', **kwargs).sel_points(theta_inc=theta, theta=theta))

    def sigmaHH(self, **kwargs):
        """Return HH backscattering coefficient. any parameter can be added to slice the results (e.g. frequency=37e9). See xarray slicing with sel method (to document)"""
        theta = np.array(self.data.theta)
        return 4*np.pi*np.cos(np.radians(theta))*_strongsqueeze(self.data.sel(polarization_inc='H', polarization='H', **kwargs).sel_points(theta_inc=theta, theta=theta))

    def sigmaHV(self, **kwargs):
        """Return HV backscattering coefficient. any parameter can be added to slice the results (e.g. frequency=37e9). See xarray slicing with sel method (to document)"""
        theta = np.array(self.data.theta)
        return 4*np.pi*np.cos(np.radians(theta))*_strongsqueeze(self.data.sel(polarization_inc='H', polarization='V', **kwargs).sel_points(theta_inc=theta, theta=theta))

    def sigmaVH(self, **kwargs):
        """Return VH backscattering coefficient. any parameter can be added to slice the results (e.g. frequency=37e9). See xarray slicing with sel method (to document)"""
        theta = np.array(self.data.theta)
        return 4*np.pi*np.cos(np.radians(theta))*_strongsqueeze(self.data.sel(polarization_inc='V', polarization='H', **kwargs).sel_points(theta_inc=theta, theta=theta))

    def save(self, filename):
        """save a result to disk. Under the hood, this is a netCDF file produced by xarray (http://xarray.pydata.org/en/stable/io.html)."""
        self.data.to_netcdf(filename)
    #def groupby(self, variable):
    #    """iterated over a given variable. Variable is typically frequency, theta, polarization or snowpack"""
    #
    #    return ResultGroup(self.data.groupby(variable))
    #    #for x, data in self.data.groupby(variable):
    #    #    yield Result(data)


# DON'T ERASE THIS, this is not needed at this stage but could be.
# This is ResultGroup is inspired from xarray, itself being inspired from pandas
# There is probably a few rough corner, but it works.
# The idea is to have the syntax: result.groupby(variabletoselect).methodinresult
# to work. For this we implement the apply method which work with any function
# and make it work with method from the Result class (or any class that).
# The injection of method is autmatic, only the name of function to inject is manual (whitelist principle)
#

# class ResultGroup(object):

#     def __init__(self, group):
#         self.group = group

#     def __iter__(self):
#         return iter(self.group)

#     def apply(self, func, **kwargs):
#         """Apply a function over each result in the group and concatenate them
#         together into a new array.

#         **kwargs
#             Used to call `func(ar, **kwargs)` for each result.
#         """

#         # apply func with optional argument to every xarray. Return a list... is the best ?
#         return [func(Result(data), **kwargs) for i, data in self.group]

#     @classmethod
#     def _apply_method(cls, func):
#         # return a method that execute apply to the function func.
#         def wrapped_func(self, **kwargs):
#             return self.apply(func, **kwargs)
#         return wrapped_func


# # inject apply method
# def inject_apply_methods(cls, methods):
#     for name in methods:
#         func = cls._apply_method(getattr(Result, name))
#         func.__name__ = name
#         #func.__doc__ = _REDUCE_DOCSTRING_TEMPLATE.format(
#         #    name=name, cls=cls.__name__,
#         #    extra_args=cls._reduce_extra_args_docstring)
#         setattr(cls, name, func)
#
# inject_apply_methods(ResultGroup, ['TbV', 'TbH', 'polarization_ratio', 'sigmaVV', 'sigmaHH', 'sigmaHV', 'sigmaVH'])
# END OF DON'T ERASE


def concat_results(result_list, coord):
    """Concatenate several results from :py:meth:`smrt.core.model.Model.run` (of type :py:class:`Result`) into a single result (of type :py:class:`Result`). This extends
    the number of dimension in the xarray hold by the instance. The new dimension is specified with coord

    :param result_list: list of results returned by :py:meth:`smrt.core.model.Model.run` or other functions.
    :param coord: a tuple (dimension_name, dimension_values) for the new dimension. Dimension_values must be a sequence or array with the same length as result_list.

    :returns: :py:class:`Result` instance

    """

    dim_name, dim_value = coord

    return Result(xr.concat([result.data for result in result_list], pd.Index(dim_value, name=dim_name)))


def _strongsqueeze(x):
    # TODO improve this to be optional using a global or a Result attribute...

    x = x.squeeze()
    if x.size == 1:
        return float(x)
    else:
        return x
