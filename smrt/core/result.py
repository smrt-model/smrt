# coding: utf-8

""" The results of RT Solver are hold by the :py:class:`Result` class. This class provides several functions
to access to the Stokes Vector and Muller matrix in a simple way. Most notable ones are :py:meth:`Result.TbV` and :py:meth:`Result.TbH`
for the passive mode calculations and :py:meth:`Result.sigmaHH` and :py:meth:`Result.sigmaVV`. :py:meth:`Result.to_dataframe` is also 
very convenient for the sensors with a channel map (all specific satellite sensors have such a map, 
only generic sensors as :py:meth:`smrt.sensor_list.active` and :py:meth:`smrt.sensor_list.passive` does not provide a map by default).

In addition, the RT Solver stores some information in Result.other_data. Currently this includes the effective_permittivity,
ks and ka for each layer. The data are accessed directly with e.g. result.other_data['ks'].

To save results of calculations in a file, simply use the pickle module or other serialization schemes. We may provide a unified and
inter-operable solution in the future.

Under the hood, :py:class:`Result` uses xarray module which provides multi-dimensional array with explicit, named, dimensions. Here the
common dimensions are frequency, polarization, polarization_inc, theta_inc, theta, and phi. They are created by the RT Solver. The interest
 of using named dimension is that slice of the xarray (i.e. results) can be selected based on the dimension name whereas with numpy the order
 of the dimensions matters. Because this is very convenient, users may be interested in adding other dimensions specific to their context such
  as time, longitude, latitude, points, ... To do so, :py:meth:`smrt.core.model.Model.run` accepts a list of snowpack and optionally
  the parameter `snowpack_dimension` is used to specify the name and values of the new dimension to build.

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
from smrt.utils import dB
from smrt.core.error import SMRTError
from smrt.core import lib


def open_result(filename):
    """read a result save to disk. See :py:meth:`Result.save` method."""
    data = xr.open_dataarray(filename, autoclose=True)

    #  argh... need to convert polarization in unicode!
    for d in data.dims:
        if d.startswith("polarization"):
            data[d] = data[d].astype("U1")

    mode = getattr(data.attrs, 'mode', None)
    if (mode is None) or (mode not in 'AP'):
        # guess the mode
        if 'theta_inc' in data.coords:
            mode = 'A'
        else:
            mode = 'P'

    return make_result(mode, data)


def make_result(sensor, *args, **kwargs):
    """create an active or passive result object according to the mode"""

    if sensor.mode == 'A':
        return ActiveResult(*args, channel_map=sensor.channel_map, **kwargs)
    else:
        return PassiveResult(*args, channel_map=sensor.channel_map, **kwargs)


class Result(object):
    """ Contains the results of a/many computations and provides convenience functions to access these results

    """

    def __init__(self, intensity, coords=None, channel_map=None, other_data={}, mother_df=None):
        """Construct results array with the given intensity array (numpy array or xarray) and dimensions if numpy array is given

"""
        super().__init__()

        if isinstance(intensity, xr.DataArray):
            self.data = intensity
        else:
            self.data = xr.DataArray(intensity, coords)

        for d in other_data.values():
            assert isinstance(d, xr.DataArray)  # this is emitter responsability to precise the coordinates
        self.other_data = other_data

        # a dataframe can be provided and will be merged with the results when using return_as_dataframe or to_dataframe
        self.mother_df = mother_df

        if hasattr(self, "mode"):
            self.data.attrs['mode'] = self.mode
        else:
            raise SMRTError("Result base class is abstract, uses a subclass instead. The subclass must define the 'mode' attribute")

        self.channel_map = channel_map or dict()

    @property
    def coords(self):
        """Return the coordinates of the result (theta, frequency, ...). Note that the coordinates are also result attribute,
        so result.frequency works (and so on for all the coordinates)."""
        return self.data.coords

    def __getattr__(self, attr):
        if attr != "data" and attr in self.data.coords:
            return self.data.coords[attr]
        else:
            raise AttributeError("AttributeError: '%s' object has no attribute '%s'" % (type(self), attr))

    def save(self, filename):
        """save a result to disk. Under the hood, this is a netCDF file produced by xarray (http://xarray.pydata.org/en/stable/io.html)."""
        self.data.to_netcdf(filename)

    def sel_data(self, channel=None, **kwargs):
        raise NotImplementedError("must be implemented in a subclass")

    def return_as_dataframe(self, name, channel_axis=None, **kwargs):
        # is to be called from child classes called to_dataframe

        def xr_to_dataframe(x, name):
            # workaround for when the resulting array has no dims anymore
            if x.dims:
                return x.to_dataframe(name=name)
            else:
                return pd.DataFrame([float(x)], columns=[name])

        if channel_axis in ["column", "index"]:
            if not self.channel_map:
                raise SMRTError("No channel information is given in the result. Unable to index the result by channel.")

            # concat the dataframe obtained for each channel
            df = pd.concat([xr_to_dataframe(self.sel_data(channel=ch, **kwargs), name=ch)
                            for ch in self.channel_map],
                           axis=1, join='inner')

            if channel_axis == "index":
                droplevel = not df.index.name and len(df.index) == 1 and df.index[0] == 0  # this is our added index, remove it
                df = df.stack()
                if isinstance(df, pd.Series):
                    df = pd.DataFrame(df, columns=[name])

                df.index.set_names('channel', level=-1)
                if droplevel:
                    df = df.droplevel(0)
        elif channel_axis is None:
            df = xr_to_dataframe(self.sel_data(**kwargs), name=name)
        else:
            raise SMRTError('channel_axis argument must be None, "column" or "index"')

        if self.mother_df is not None:
            if channel_axis == "column":
                # join without alignment. We assume both have the same order. In principle this is the case with model.py
                df = df.reset_index(drop=True).join(self.mother_df.reset_index(drop=True))
                df.index = self.mother_df.index
            elif channel_axis is None:

                # df is multiindex by construction
                assert isinstance(df.index, pd.MultiIndex)
                # join, assuming the index is unique # add a check
                if not self.mother_df.index.is_unique:
                    raise SMRTError("The index of the snowpack DataFrame in input of Model.run "
                                    "must be unique for calling to_dataframe. "
                                    "The index is used to join the result and original DataFrame.")
                name = self.mother_df.index.names
                if name[0] is None:
                    # give a name to the mother_df for the join
                    name = df.index.names[0]
                    if name in df.columns:
                        raise SMRTError("The index of the snowpack DataFrame in input of Model.run "
                                        "shall be named to avoid naming conflict in to_dataframe.")
                    mother_df = self.mother_df.copy()
                    mother_df.index.name = name
                else:
                    mother_df = self.mother_df

                df = df.reset_index().join(mother_df, on=name).set_index(df.index.names)

            # silently ignore the case with channel_axis='index'. It is not clear what should be done but most probably nothing.
            # for this reason, we don't any raise exception or warning.

            # warnings("running a model with a pandas DataFrame snowpack (or Series) and calling to_dataframe with channel_axis='index' "
            #             "is ambiguous / not implemented. The result is returned without joining with the snowpack DataFrame.")

        return df

    def to_series(self, **kwargs):
        """return the result as a series with the channels defined in the sensor as index.
        This requires that the sensor has declared a channel list.

        """
        return self.return_as_dataframe('out', channel_axis='column', **kwargs).iloc[0]


class PassiveResult(Result):

    mode = 'P'

    def sel_data(self, channel=None, **kwargs):
        # this function allows selection as xarray.DataArray.sel and in addition by channel if a channel_map is defined.

        # ffilter the variables of channel_map[channel] that are effectively in self.data.dims
        # and apply them to the selector sel in addition to kwargs

        if channel is not None:
            kwargs.update({k: v for k, v in self.channel_map[channel].items() if k in self.data.dims})

        return self.data.sel(drop=True, **kwargs)

    def Tb(self, channel=None, **kwargs):
        """Return brightness temperature. Any parameter can be added to slice the results (e.g. frequency=37e9 or polarization='V').
         See xarray slicing with sel method (to document). It is also posisble to select by channel if the sensor has a channel_map.

         :param channel: channel to select
         :param **kwargs: any parameter to slice the results.
"""
        return _strongsqueeze(self.sel_data(channel=channel, **kwargs))

    def Tb_as_dataframe(self, channel_axis=None, **kwargs):
        """See :py:meth:`PassiveResult`.to_dataframe
"""

        return self.to_dataframe(channel_axis=None, **kwargs)

    def to_dataframe(self, channel_axis="auto", **kwargs):
        """Return brightness temperature as a pandas.DataFrame. Any parameter can be added to slice the results
        (e.g. frequency=37e9 or polarization='V'). See xarray slicing with sel method (to document).
        In addition channel_axis controls the format of the output. If set to None, the DataFrame has a multi-index with all the
        dimensions (frequency, polarization, ...).
        If channel_axis is set to "column", and if the sensor has a channel map, the channels are
        in columns and the other dimensions are in index. If set to "index", the channel are in index with all the other dimensions.

        The most conviennent is probably channel_axis="column" while channel_axis=None (default) contains all the data even those
        not corresponding to a channel and applies to any sensor even those without channel_map. If set to "auto", 
        the channel_axis is "column" if the channel map exit, otherwise is None.

        :param channel_axis: controls whether to use the sensor channel or not and if yes, as a column or index.
"""
        if channel_axis == "auto":
            channel_axis = "column" if self.channel_map else None

        return super().return_as_dataframe(name='Tb', channel_axis=channel_axis, **kwargs)

    def TbV(self, **kwargs):
        """Return V polarization. Any parameter can be added to slice the results (e.g. frequency=37e9).
         See xarray slicing with sel method (to document)"""
        return _strongsqueeze(self.data.sel(polarization='V', **kwargs))

    def TbH(self, **kwargs):
        """Return H polarization. Any parameter can be added to slice the results (e.g. frequency=37e9).
         See xarray slicing with sel method (to document)"""
        return _strongsqueeze(self.data.sel(polarization='H', **kwargs))

    def polarization_ratio(self, ratio="H_V", **kwargs):
        """Return polarization ratio. Any parameter can be added to slice the results (e.g. frequency=37e9).
         See xarray slicing with sel method (to document)"""
        return _strongsqueeze(self.data.sel(polarization=ratio[0], **kwargs) / self.data.sel(polarization=ratio[-1], **kwargs))


class ActiveResult(Result):

    mode = 'A'

    def sel_data(self, channel=None, return_backscatter=False, **kwargs):
        # this function allows selection as xarray.DataArray.sel and in addition by channel if a channel_map is defined.

        # ffilter the variables of channel_map[channel] that are effectively in self.data.dims
        # and apply them to the selector sel in addition to kwargs

        if channel is not None:
            kwargs.update({k: v for k, v in self.channel_map[channel].items() if k in self.data.dims})

        if return_backscatter:
            # get theta
            theta = kwargs.pop('theta', None)
            theta_inc = kwargs.pop('theta_inc', None)

            if theta is not None and theta_inc is not None:
                if not np.all(theta_inc == theta):
                    raise SMRTError('theta and theta_inc must be the same when returning backscatter')

            if theta is None:
                theta = theta_inc

            if theta is None:
                theta = self.data.theta_inc

            def select_theta(x, theta, **kwargs):
                # select by theta and deal with cases where theta is in the coords or not
                if 'theta' in x.coords:
                    return x.sel(theta=theta, theta_inc=theta, **kwargs)
                else:
                    return x.sel(theta_inc=theta, **kwargs)

            if lib.is_sequence(theta):
                # now select all the theta if it is a sequence
                x = xr.concat([select_theta(self.data, t, drop=True, **kwargs) for t in theta],
                              pd.Index(theta, name='theta_inc'))
            else:
                x = select_theta(self.data, theta, drop=True, **kwargs)

        else:
            x = self.data.sel(drop=True, **kwargs)

        if return_backscatter:
            x = (4 * np.pi * np.cos(np.deg2rad(theta))) * x
            return dB(x) if return_backscatter == "dB" else x
        else:
            return x

    def sigma(self, channel=None, **kwargs):
        """Return backscattering coefficient. Any parameter can be added to slice the results (e.g. frequency=37e9 or polarization='V').
         See xarray slicing with sel method (to document). It is also posisble to select by channel if the sensor has a channel_map.

         :param channel: channel to select
         :param **kwargs: any parameter to slice the results.
"""

        return _strongsqueeze(self.sel_data(channel=channel, return_backscatter="natural", **kwargs))

    def sigma_dB(self, channel=None, **kwargs):
        """Return backscattering coefficient. Any parameter can be added to slice the results (e.g. frequency=37e9,
        polarization_inc='V', polarization='V'). See xarray slicing with sel method (to document)
"""

        return _strongsqueeze(self.sel_data(channel=channel, return_backscatter="dB", **kwargs))

    def sigma_as_dataframe(self, channel_axis=None, **kwargs):
        """Return backscattering coefficient as a pandas.DataFrame. Any parameter can be added to slice the results
        (e.g. frequency=37e9 or polarization='V'). See xarray slicing with sel method (to document).
        In addition channel_axis controls the format of the output. If set to None, the DataFrame has a multi-index formed with all the
        dimensions (frequency, polarization, ...).
        If channel_axis is set to "column", and if the sensor has named channels (channel_map in SMRT wording), the channel are
        in columns and the other dimensions are in index. If set to "index", the channel are in index with all the other dimensions.

        The most conviennent is probably channel_axis="column" while channel_axis=None (default) contains all the data even those
        not corresponding to a channel and applies to any sensor even those without channel_map.

        :param channel_axis: controls whether to use the sensor channel or not and if yes, as a column or index.
"""

        return super().return_as_dataframe(name='sigma', channel_axis=channel_axis, return_backscatter="natural", **kwargs)

    def sigma_dB_as_dataframe(self, channel_axis=None, **kwargs):
        """See :py:meth:`ActiveResult`.to_dataframe
"""
        return self.to_dataframe(channel_axis=channel_axis, **kwargs)

    def to_dataframe(self, channel_axis=None, **kwargs):
        """Return backscattering coefficient in dB as a pandas.DataFrame. Any parameter can be added to slice the results
        (e.g. frequency=37e9 or polarization='V'). See xarray slicing with sel method (to document).
        In addition channel_axis controls the format of the output. If set to None, the DataFrame has a multi-index with all the
        dimensions (frequency, polarization, ...).
        If channel_axis is set to "column", and if the sensor has named channels (channel_map in SMRT wording), the channel are
        in columns and the other dimensions are in index. If set to "index", the channel are in index with all the other dimensions.

        If channel_axis is set to "column", and if the sensor has a channel map, the channels are
        in columns and the other dimensions are in index. If set to "index", the channel are in index with all the other dimensions.

        The most conviennent is probably channel_axis="column" while channel_axis=None (default) contains all the data even those
        not corresponding to a channel and applies to any sensor even those without channel_map. If set to "auto", 
        the channel_axis is "column" if the channel map exit, otherwise is None.

        :param channel_axis: controls whether to use the sensor channel or not and if yes, as a column or index.
"""
        if channel_axis == "auto":
            channel_axis = "column" if self.channel_map else None
        return super().return_as_dataframe(name='sigma', channel_axis=channel_axis, return_backscatter="dB", **kwargs)

    def to_series(self, **kwargs):
        """return backscattering coefficients in dB as a series with the channels defined in the sensor as index.
        This requires that the sensor has declared a channel list.

        """
        return super().to_series(return_backscatter="dB", **kwargs)

    def sigmaVV(self, **kwargs):
        """Return VV backscattering coefficient. Any parameter can be added to slice the results (e.g. frequency=37e9).
         See xarray slicing with sel method (to document)"""
        return self.sigma(polarization_inc='V', polarization='V', **kwargs)

    def sigmaVV_dB(self, **kwargs):
        """Return VV backscattering coefficient in dB. Any parameter can be added to slice the results (e.g. frequency=37e9).
         See xarray slicing with sel method (to document)"""
        return dB(self.sigmaVV(**kwargs))

    def sigmaHH(self, **kwargs):
        """Return HH backscattering coefficient. Any parameter can be added to slice the results (e.g. frequency=37e9).
         See xarray slicing with sel method (to document)"""
        return self.sigma(polarization_inc='H', polarization='H', **kwargs)

    def sigmaHH_dB(self, **kwargs):
        """Return HH backscattering coefficient in dB. Any parameter can be added to slice the results (e.g. frequency=37e9).
         See xarray slicing with sel method (to document)"""
        return dB(self.sigmaHH(**kwargs))

    def sigmaHV(self, **kwargs):
        """Return HV backscattering coefficient. Any parameter can be added to slice the results (e.g. frequency=37e9).
         See xarray slicing with sel method (to document)"""
        return self.sigma(polarization_inc='H', polarization='V', **kwargs)

    def sigmaHV_dB(self, **kwargs):
        """Return HV backscattering coefficient in dB. Any parameter can be added to slice the results (e.g. frequency=37e9).
         See xarray slicing with sel method (to document)"""
        return dB(self.sigmaHV(**kwargs))

    def sigmaVH(self, **kwargs):
        """Return VH backscattering coefficient. Any parameter can be added to slice the results (e.g. frequency=37e9).
         See xarray slicing with sel method (to document)"""
        return self.sigma(polarization_inc='V', polarization='H', **kwargs)

    def sigmaVH_dB(self, **kwargs):
        """Return VH backscattering coefficient in dB. Any parameter can be added to slice the results (e.g. frequency=37e9).
         See xarray slicing with sel method (to document)"""
        return dB(self.sigmaVH(**kwargs))

    # def groupby(self, variable):
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
    """Concatenate several results from :py:meth:`smrt.core.model.Model.run` (of type :py:class:`Result`) into a single result
    (of type :py:class:`Result`). This extends the number of dimension in the xarray hold by the instance. The new dimension
    is specified with coord

    :param result_list: list of results returned by :py:meth:`smrt.core.model.Model.run` or other functions.
    :param coord: a tuple (dimension_name, dimension_values) for the new dimension. Dimension_values must be a sequence or
    array with the same length as result_list.

    :returns: :py:class:`Result` instance

    """

    if isinstance(coord, tuple):
        dim_name, dim_value = coord

        index = pd.Index(dim_value, name=dim_name)
    elif isinstance(coord, pd.Index):
        index = coord
        if index.name is None:
            index.name = 'snowpack_index'  # hope this will not conflict with an existing column
    else:
        raise SMRTError('unknown type for the coord argument')

    ResultClass = type(result_list[0])
    if not all([type(result) == ResultClass for result in result_list]):
        raise SMRTError("The results are not all of the same type")

    # channel_map ?
    if any((res.channel_map != result_list[0].channel_map for res in result_list)):
        assert isinstance(coord, tuple)
        # different channel maps, it means we have different sensors. Merge de sensor maps.
        channel_map = {ch: dict(**r.channel_map[ch], dim_name=dv) for r, dv in zip(result_list, dim_value) for ch in r.channel_map}
    else:
        # all the channel maps are the same
        channel_map = result_list[0].channel_map

    data = xr.concat([result.data for result in result_list], index)
    other_data = {v: xr.concat([result.other_data[v] for result in result_list], index) for v in result_list[0].other_data}

    return ResultClass(data,
                       channel_map=channel_map,
                       other_data=other_data)


def _strongsqueeze(x):
    # TODO improve this to be optional using a global or a Result attribute...

    x = x.squeeze()
    if x.size == 1:
        return float(x)
    else:
        return x
