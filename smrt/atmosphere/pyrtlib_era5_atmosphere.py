# coding: utf-8
"""A non-scattering atmosphere provided by PyRTLib for SMRT using ERA5 data as input

This atmosphere is a special case using ERA5. Please refer to the general documentation `py:module::~smrt.atmosphere.pyrtlib_atmosphere`. 

"""

# Stdlib import
import os
import tempfile
from datetime import datetime
from typing import Optional
from warnings import warn

# other import
import numpy as np
import pandas as pd
import xarray as xr

from pyrtlib.apiwebservices import ERA5Reanalysis

# local import
from .pyrtlib_atmosphere import PyRTlibAtmosphereBase
from pyrtlib.utils import kgkg_to_kgm3


class PyRTlibERA5Atmosphere(PyRTlibAtmosphereBase):

    def __init__(self, longitude,
                 latitude,
                 date,
                 datafile=None,
                 use_grib=True,
                 era5_directory=None,
                 absorption_model=None):
        super().__init__(absorption_model=absorption_model)

        if use_grib:
            extension = "grib"
            ERA5cls = _ERA5Reanalysis_with_grib
        else:
            extension = "nc"
            ERA5cls = ERA5Reanalysis

        if datafile is None:
            if era5_directory is None:
                era5_directory = tempfile.gettempdir()

            extension = "grib" if use_grib else "nc"

            datafile_name = 'era5_reanalysis-{}.'.format(date.isoformat()) + extension  # from pyrtlib
            datafile = os.path.join(era5_directory, datafile_name)
            # rename with lat, lon included to avoid errors
            new_datafile = datafile[:-len(extension)] + f'-{longitude:.1f}-{latitude:.1f}.{extension}'

            if not os.path.exists(new_datafile):
                # automatically download the file
                warn(f"Downloading file ERA5 file: {datafile}")
                # it seems that small extents fail in cdsapi retrieval
                datafile = ERA5cls.request_data(era5_directory, date, (longitude, latitude), offset=0.4)
                assert datafile[-len(extension):] == extension, f"filename: {datafile}"
                os.rename(datafile, new_datafile)
            datafile = new_datafile

        df_era5 = ERA5cls.read_data(datafile, (longitude, latitude))
        self.df_era5 = df_era5

        self.z = df_era5.z.values
        self.p = df_era5.p.values
        self.t = df_era5.t.values
        self.rh = df_era5.rh.values

        self.cloudy = True
        self.cldh = np.empty((2, 1))
        self.cldh[:, 0] = np.array([np.min(df_era5.z), np.max(df_era5.z)])

        total_mass = 1 - df_era5.ciwc.values - df_era5.clwc.values - df_era5.crwc.values - df_era5.cswc.values

        norm = (1 / total_mass) * kgkg_to_kgm3(df_era5.q.values * (1 / total_mass), df_era5.p.values, df_era5.t.values) * 1000
        self.denice = df_era5.ciwc.values * norm
        self.denliq = df_era5.clwc.values * norm


class _ERA5Reanalysis_with_grib(ERA5Reanalysis):
    """temporary hack to retrieve ERA5 data in grib instead of netcdf which seems to be broken. The only disavantage is to require the dependency (cfgrib)
"""

    @classmethod
    def read_data(cls, file: str, lonlat: tuple) -> pd.DataFrame:
        """Read data from the ERA5 Reanalysis dataset.

        Args:
            file (str): The netcdf file
            lonlat (tuple): longitude and latitude

        Returns:
            pandas.DataFrame: Dataframe containing the variables retrieved.

        .. note:: Variables name and units information are reported within the attribute `units` of
            the returned dataframe (see example below).

        Example:
            .. code-block:: python

                >>> from pyrtlib.apiwebservices import ERA5Reanalysis
                >>> lonlat = (15.8158, 38.2663)
                >>> date = datetime(2020, 2, 22, 12)
                >>> nc_file = ERA5Reanalysis.request_data(tempfile.gettempdir(), date, lonlat)
                >>> df_era5 = ERA5Reanalysis.read_data(nc_file, lonlat)
                >>> df_era5.attrs['units']
                {'p': 'hPa',
                'z': 'km',
                't': 'K',
                'rh': '%',
                'clwc': 'kg kg-1',
                'ciwc': 'kg kg-1',
                'crwc': 'kg kg-1',
                'cswc': 'kg kg-1',
                'o3': 'kg kg-1',
                'q': 'kg kg-1'}

        .. note:: To convert specific cloud water content (CLWC) or specific cloud ice water content (CIWC) 
            from kg kg-1 to g m-3 using this function :py:meth:`pyrtlib.utils.kgkg_to_gm3`
        """
        import cfgrib
        from pyrtlib.utils import atmospheric_tickness

        ERAFIVE = cls()
        data = xr.open_dataset(file, engine='cfgrib')
        lats = data['latitude'].values
        lons = data['longitude'].values
        idx, _ = ERAFIVE._find_nearest(lons, lats, lonlat)
        idx = dict(latitude=lats[idx], longitude=lons[idx])

        pres = data['isobaricInhPa'].values
        temp = data['t'].sel(**idx).values
        # RH in decimal
        rh = data['r'].sel(**idx).values / 100
        clwc = data['clwc'].sel(**idx).values
        ciwc = data['ciwc'].sel(**idx).values
        crwc = data['crwc'].sel(**idx).values
        cswc = data['cswc'].sel(**idx).values
        ozone = data['o3'].sel(**idx).values
        q = data['q'].sel(**idx).values

        print(pres, temp, q)
        z = atmospheric_tickness(pres, temp, q)  # Altitude in km
        print(z)

        date = data['time'].values

        df = pd.DataFrame({'p': pres,
                           'z': z,
                           't': temp,
                           'rh': rh,
                           'clwc': clwc,
                           'ciwc': ciwc,
                           'crwc': crwc,
                           'cswc': cswc,
                           'o3': ozone,
                           'q': q,
                           'time': np.repeat(date, len(pres))
                           })
        df.attrs['units'] = {'p': 'hPa',
                             'z': 'km',
                             't': 'K',
                             'rh': '%',
                             'clwc': 'kg kg-1',
                             'ciwc': 'kg kg-1',
                             'crwc': 'kg kg-1',
                             'cswc': 'kg kg-1',
                             'o3': 'kg kg-1',
                             'q': 'kg kg-1'}

        return df

    @staticmethod
    def request_data(path: str, time: datetime,
                     lonlat: tuple,
                     resolution: Optional[float] = 0.25,
                     offset: Optional[float] = 0.4) -> str:
        """Download ERA5Reanalysis data from the Copernicus Climate Change Service using the grib format as the netcdf
        format seems to be broken at the moment (April 2024).

        Args:
            path (str): The output directory
            time (datetime): The date and time of the desired observation.
            lonlat (tuple): The coordinatre in degrees, longitude and latitude
            resolution (Optional[float], optional): The pixel size of the requested grid data. Defaults to 0.25.
            offset (Optional[float], optional): The offset to apply to coordinates to get the extent. Defaults to 0.3.

        Returns:
            str: The path to downloaded netcdf file
        """

        import cdsapi

        # North, West, South, Est
        extent = [lonlat[1] + offset, lonlat[0] - offset,
                  lonlat[1] - offset, lonlat[0] + offset]
        grib_file_name = 'era5_reanalysis-{}.grib'.format(time.isoformat())
        grib_file = os.path.join(path, grib_file_name)

        variables = ['relative_humidity', 'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
                     'specific_humidity', 'specific_rain_water_content', 'specific_snow_water_content', 'ozone_mass_mixing_ratio', 'temperature']
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': variables,
                'pressure_level': [
                    '1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150',
                    '175', '200', '225', '250', '300', '350', '400', '450', '500', '550', '600',
                    '650', '700', '750', '775', '800', '825', '850', '875', '900', '925', '950',
                    '975', '1000',
                ],
                'year': time.year,
                'month': time.month,
                'day': time.day,
                'time': '{}:00'.format(time.hour),
                'area': extent,
                'grid': [resolution, resolution],
                'format': 'grib',
            },
            grib_file)

        return grib_file
