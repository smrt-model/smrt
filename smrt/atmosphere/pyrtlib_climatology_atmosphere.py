# coding: utf-8
"""A non-scattering atmosphere provided by PyRTLib for SMRT using climatology data as input

This atmosphere is a special case using climatology. Please refer to the general documentation `py:module::~smrt.atmosphere.pyrtlib_atmosphere`. 

"""

# Stdlib import

# other import
from pyrtlib.climatology.atmospheric_profiles import AtmosphericProfiles as atmp
from pyrtlib.utils import ppmv2gkg, mr2rh

# local import
from .pyrtlib_atmosphere import PyRTlibAtmosphereBase
from ..core.error import SMRTError

class PyRTlibClimatologyAtmosphere(PyRTlibAtmosphereBase):


    def __init__(self, profile='Subarctic Summer', absorption_model=None):
        super().__init__(absorption_model=absorption_model)

        if isinstance(profile, str):
            for k, v in atmp.atm_profiles().items():
                if v == profile:
                    profile = k
                    break
            else:
                raise SMRTError(f"The requested atmospheric profile '{profile}' isn't among the available profiles: {", ".join(atmp.atm_profiles().values())}")

        self.z, self.p, d, self.t, md = atmp.gl_atm(profile)
        gkg = ppmv2gkg(md[:, atmp.H2O], atmp.H2O)
        self.rh = mr2rh(self.p, self.t, gkg)[0] / 100
