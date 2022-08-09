# coding: utf-8

""" Compute Strong Fluctuation Theory scattering. This theory requires the scatterers to be smaller than the wavelength

This model is only compatible with the Exponential autocorrelation function only

"""

import numpy as np

from ..core.error import SMRTError
from ..core.globalconstants import C_SPEED
from ..permittivity.generic_mixing_formula import polder_van_santen
from .rayleigh import Rayleigh


class SFT_Rayleigh(Rayleigh):
    """
    """

    def __init__(self, sensor, layer):

        # super().__init__()  # must not be called. Todo: write a generic RayleighBase object with phase function methods only

        # check here the limit of the Rayleigh model

        f = layer.frac_volume

        eb = layer.permittivity(0, sensor.frequency)  # background permittivity
        es = layer.permittivity(1, sensor.frequency)  # scatterer permittivity
        e0 = 1  # always

        lmda = C_SPEED / sensor.frequency
        k0 = 2 * np.pi / lmda * np.sqrt(e0)

        corr_length = layer.microstructure.corr_length

        self._effective_permittivity = polder_van_santen(f, eb, es)
        eg = self._effective_permittivity  # short
        kg = k0 * np.sqrt(eg / e0)

        delta = 9 * eg**2 / e0**2 * (f * ((es - eg) / (es + 2 * eg))**2 + (1 - f) * ((eb - eg) / (eb + 2 * eg))**2)

        beta = 1 / corr_length - 1j * kg

        I1 = 1 / (beta**2 + kg**2)
        I2 = -3.0 / 2 * beta / kg**2 + 1.0 / (2 * kg) * (3 * beta**2 / kg**2 + 1) * np.arctan(kg / beta)
        I3 = 3 / kg**2 - 1 / (beta**2 + kg**2) - 3 * beta / kg**3 * np.arctan(kg / beta)
        I4 = 1.0 / 3 + beta**2 / (2 * kg**2) - beta / (2 * kg) * (beta**2 / kg**2 + 1) * np.arctan(kg / beta)

        Eeff = eg + k0**2 * delta * (2 * I1 / 3 - 1j * I2 / kg - I3 / 3 + I4 / (k0**2 * eg))

        self.ka = 2 * k0 * np.sqrt(eg).imag
        self.ks = 2 * k0 * np.sqrt(Eeff).imag - self.ka
