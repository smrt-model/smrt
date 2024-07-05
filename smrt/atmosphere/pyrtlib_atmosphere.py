# coding: utf-8
"""A non-scattering atmosphere provided by PyRTLib for SMRT

PyRTLib is a standalone Python package for non-scattering line-by-line microwave radiative transfer simulations
of the atmosphere emission (passive microwave) developed by Salvatore Larosa, Domenico Cimini, Donatello Gallucci,
Saverio Teodosio Nilo, and Filomena Romano. According to the authors, it does not intent to compete with state-of-the-
art models as used in the meteorological centers for instance, but has instead an educational purpose and has the major
advantage of being fully written in Python, synonym of easy installation and compatibility with SMRT.

PyRTLib is licensed under the GPL-3.0 License, and it must be installed independently of SMRT. pip install smrt[pyrtlib] may work.

PyRTlib allows to simulate and calculate radiometric parameters and estimating propogation parameters needed by SMRT using
meteorological data as input. Some meteorological datasets are built-in in PyRTlib and others can be download and used
directly in PyRTlib. Available datasets are described on the main website: https://satclop.github.io/pyrtlib/

Citation:
Larosa, S., Cimini, D., Gallucci, D., Nilo, S. T., and Romano, F.: PyRTlib: an educational Python-based library for non-
scattering atmospheric microwave radiative transfer computations, Geosci. Model Dev., 17, 2053–2076,
https://doi.org/10.5194/gmd-17-2053-2024, 2024.

To build an atmosphere in general, it is recommended to use the helper function
:py:func:`~smrt.inputs.make_model.make_atmosphere`. In the case of PyRTLib, there are in fact three ways to initialize the
atmosphere depending on the atmospheric input data to be used.

The simpliest is for climatological profiles::

    from smrt import make_atmosphere

    atmos = make_atmosphere('pyrtlib_climatology_atmosphere', profile='Subarctic Summer', absorption_model = 'R20')

The list of climatologies is however limited, see the documentation at https://satclop.github.io/pyrtlib/en/main/generated/pyrtlib.climatology.AtmosphericProfiles.html

For a more specific calculations in term of location and date, it is possible obtain data from ERA5 Reanalysis::

    from smrt import make_atmosphere

    atmos = make_atmosphere('pyrtlib_era5_atmosphere', longitude=-75.07, latitude=123., date=datetime(2020, 2, 22, 12), absorption_model = 'R20')

An ERA5 file will be automatically downloaded which requires the installation of the CDSAPI and cfgrib python packages and to obain a CDS API Key.
Please follow the instructions on the Copernicus site: https://cds.climate.copernicus.eu/api-how-to . 
Note that in April 2024, the CDS is announced to be disrupted "soon", which will impose changes in SMRT.

The downloaded file is copied in a temporary directory, unless the `era5_directory` argument is specified, which is
recommended to avoid repetitive downloads.

If interested in several locations, it is more efficient to download a single file with the full extent following the PyRTlib documentation:
https://satclop.github.io/pyrtlib/en/main/generated/pyrtlib.apiwebservices.ERA5Reanalysis.request_data.html and then use the 'ncfile' argument::

    from smrt import make_atmosphere

    atmos = make_atmosphere('pyrtlib_era5_atmosphere', ncfile='era5_reanalysis-2023-05-16T18:00:00.nc',
                                                       longitude=-75.07, latitude=123., date=datetime(2020, 2, 22, 12),
                                                       absorption_model = 'R20')


PyRTlib includes many absorption models and the list can be obtained using::

    PyRTlibAtmosphere.available_absorption_models()



"""

# Stdlib import
import functools

# other import
import numpy as np

from pyrtlib.tb_spectrum import TbCloudRTE
from pyrtlib.climatology.atmospheric_profiles import AtmosphericProfiles
from pyrtlib.absorption_model import AbsModel

# local import
from ..core.error import SMRTError
from ..core.atmosphere import AtmosphereBase, AtmosphereResult
from ..core.globalconstants import GHz


default_absorption_model = 'R20'


class PyRTlibAtmosphereBase(AtmosphereBase):

    def __init__(self, absorption_model=None):

        self.absorption_model = absorption_model if absorption_model is not None else default_absorption_model
        self.cloudy = False

    @classmethod
    def available_absorption_models(cls):
        return AbsModel.implemented_models()

    def run(self, frequency, costheta, npol):

        freqGHz = np.atleast_1d(frequency) / GHz
        rte = TbCloudRTE(z=self.z,
                         p=self.p,
                         t=self.t,
                         rh=self.rh,
                         frq=freqGHz,
                         angles=np.atleast_1d(90 - np.rad2deg(np.arccos(costheta))),
                         )

        rte.emissivity = np.array([0])  # The surface is taken into account in SMRT

        if self.cloudy:
            rte.cloudy = True
            rte.init_cloudy(self.cldh, self.denice, self.denliq)
        rte.init_absmdl(self.absorption_model)

        rte.satellite = True
        upwelling = rte.execute()
        rte.satellite = False
        downwelling = rte.execute()

        trans = np.exp(-(downwelling['taudry'].values + downwelling['tauwet'].values + downwelling['tauliq'].values + downwelling['tauice'].values))

        return AtmosphereResult(tb_down=np.repeat(downwelling['tbtotal'].values, npol),
                                tb_up=np.repeat(upwelling['tbtotal'].values, npol),
                                transmittance=np.repeat(trans, npol))


class PyRTlibAtmosphere(PyRTlibAtmosphereBase):

    def __init__(self,
                 altitude,
                 pressure,
                 temperature,
                 rh,
                 cloud_base_top=None,
                 ice_density=0,
                 water_density=0,
                 absorption_model=None,
                 ):
        """
        Return an PyRTlib atmosphere with a prescribed profile with pressure, temperature and humidity and optionally clouds

        :param altitude: Altitude of the layers (m). The first element of the array should be the highest.
        :param pressure: Pressure in each layers (Pa).
        :param temperature: Temperature profile (K).
        :param rh: Relative humidity profile (fraction).
        :param cloud_base_top: Tuple with cloud base and top altitude (m). It is optional.
        :param denice: Ice density profile (:math:`kg m^{-3}`).
        :param denliq: Liquid density profile (:math:`kg m^{-3`).
        :param absorption_model: An available absorption model. See
        """
        super().__init__(absorption_model=absorption_model)

        self.z = altitude / 1000  # convert to km
        self.p = pressure * 100   # convert to mbar
        self.t = temperature      # in K
        self.rh = rh

        self.denice = ice_density * 1000  # convert to g/m3
        self.denliq = water_density * 1000  # convert to g/m3

        if cloud_base_top is None:
            self.cloudy = False
        else:
            self.cloudy = True
            self.cldh = np.array(cloud_base_top) / 1000   # assemble and convert to km
