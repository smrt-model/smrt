# coding: utf-8

""" Use prescribed scattering ks and absorption ka coefficients and effective permittivity in the layer.
    The phase matrix has the Rayleigh form with prescribed scattering coefficient

    This model is compatible with any microstructure but requires that ks, ka, and optionally effective permittivity to
    be set in the layer


Example::

    m = make_model("prescribed_kskaeps", "dort")
    snowpack.layers[0].ks = ks
    snowpack.layers[0].ka = ka
    snowpack.layers[0].effective_permittivity = eff_eps

"""

import numpy as np

from .rayleigh import Rayleigh


class Prescribed_KsKaEps(Rayleigh):
    """
    """
    def __init__(self, sensor, layer):

        #e0 = layer.permittivity(0, sensor.frequency)  # background permittivity
        #eps = layer.permittivity(1, sensor.frequency)  # scatterer permittivity

        self._effective_permittivity = layer.effective_permittivity

        self.ks = layer.ks
        self.ka = layer.ka

