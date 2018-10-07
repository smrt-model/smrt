# coding: utf-8

""" This code is depreciated, please use dmrt_qcacp_shortrange or dmrt_qca_shortrange
"""


# Stdlib import
import math
import cmath

# other import
import numpy as np


# local import
from ..core.error import SMRTError
from ..core.globalconstants import C_SPEED
from .rayleigh import Rayleigh

raise SMRTError("This code is depreciated, please use dmrt_qcacp_shortrange or dmrt_qca_shortrange")

#
# DMRT short range derives from Rayleigh because it has the same phase matrix
# only the scattering and absorption coefficient are different.
#

npol = 2


class DMRT_ShortRange(Rayleigh):

    """ DMRT electromagnetic model in the short range limit (grains AND aggregates are small) as implemented in DMRTML

        :param sensor: sensor instance
        :param layer: layer instance
        :dense_snow_correction: set how snow denser than half the ice density (ie. fractional volume larger than 0.5 is handled).
        "auto" means that snow is modeled as air bubble in ice instead of ice spheres in air.
        "bridging" should be developed in the future.
    """

    def __init__(self, sensor, layer, dense_snow_correction="auto"):

        if layer.frac_volume > 0.5 and dense_snow_correction == "auto":
            layer = layer.inverted_medium()

        f = layer.frac_volume

        e0 = layer.permittivity(0, sensor.frequency)  # background permittivity
        es = layer.permittivity(1, sensor.frequency)  # scatterer permittivity

        lmda = C_SPEED / sensor.frequency

        if not hasattr(layer.microstructure, "stickiness") or not hasattr(layer.microstructure, "compute_t"):
            raise SMRTError("DMRT_ShortRange is only compatible with SHS microstructure model")

        radius = layer.microstructure.radius
        t = layer.microstructure.compute_t()

        # these formulations are taken from DMRT-ML Picard et al. 2013
        #
        # Solve the 0th-order solution: Eeff0
        # Eeff0^2 + Eeff0 *[ (Ei-eo)/3*(1-4f)-eo] - eo (Ei-1)/3*(1-f) = 0

        b = (es - e0) * (1.0 - 4.0 * f) / 3.0 - e0
        c = -e0 * (es - e0) * (1.0 - f) / 3.0

        discriminant = b**2 - 4 * c

        # solution
        Eeff0 = 0.5 * (-b + cmath.sqrt(discriminant))

        if Eeff0.real < 1:
            Eeff0 = 0.5 * (-b - cmath.sqrt(discriminant))

        # Solve 1st-order solution: E
        Eeff = e0 + (Eeff0 - e0) * (1 + 2.0j / 9.0 * (2 * math.pi * radius / lmda)**3 *
                                    cmath.sqrt(Eeff0) * (es - e0) / (1.0 + (es - e0) / (3 * Eeff0) * (1.0 - f)) *
                                    (1.0 - f)**4 / (1.0 + 2 * f - t * f * (1.0 - f))**2)

        albedo = 2.0 / 9.0 * (2 * np.pi * radius / lmda)**3 * f / (2 * cmath.sqrt(Eeff).imag) *  \
            abs((es - e0) / (1 + (es - e0) / (3 * Eeff0) * (1.0 - f)))**2 * \
            (1.0 - f)**4 / (1.0 + 2 * f - t * f * (1.0 - f))**2
        
        if albedo >= 1:
            raise SMRTError("Grain diameter is too large for DMRT_ShortRange resulting in single scattering albedo larger than 1."
                            "It is recommended to decrease the size or used an alternative emmodel able to do Mie calculations.")

        beta = 2 * math.pi / lmda * 2 * cmath.sqrt(Eeff).imag

        self._effective_permittivity = Eeff
        self.ks = albedo * beta
        self.ka = beta - self.ks

    def basic_check(self):
        # TODO Ghi: check the microstructure model is compatible.
        # if we want to be strict, only IndependentShpere should be valid, but in pratice any
        # model of sphere with a radius can make it!
        if not hasattr(self.layer.microstructure, "radius"):
            raise SMRTError("Only microstructure_model which defined a `radius` can be used with Rayleigh scattering")

    # The phase function is inherited from Rayleigh  // Don't remove the commented code
    #    def phase(self, m, mhu):

    # The ke function is inherited from Rayleigh  // Don't remove the commented code
    # def ke(self, mhu):
    #    return np.full(2*len(mhu), self.ks+self.ka)

    # The effective_permittivity is inherited from Rayleigh  // Don't remove the commented code
    # def effective_permittivity(self):
    #    return self._effective_permittivity
