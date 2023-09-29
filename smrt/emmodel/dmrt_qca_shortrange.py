# coding: utf-8

""" Compute scattering with DMRT QCA Short range. Short range means that it is accurate only for small
and weakly sticky spheres (high stickiness value). It diverges (increasing scattering coefficient) if these conditions
are not met. Numerically the size conditions can be evaluated with the ratio radius/wavelength as for Rayleigh scatterers.
For the stickiness, it is more difficult as this depends on the size of the scatterers and the fractional volume. In any case, it is
dangerous to use too small a stickiness value, especially if the grains are big.

This model is only compatible with the SHS microstructure model.

Examples:

    from smrt import make_snowpack, make_sensor

    density = [345.0]
    temperature = [260.0]
    thickness = [70]
    radius = [750e-6]
    stickiness = [0.1]

    snowpack = make_snowpack(thickness, "sticky_hard_spheres",
                        density=density, temperature=temperature, radius=radius, stickiness=stickiness)


    # create the EM Model - Equivalent DMRTML
    m = make_model("dmrt_shortrange", "dort")

    # create the sensor
    theta = np.arange(5.0, 80.0, 5.0)
    radiometer = sensor.amsre()


"""

# Stdlib import
import math
import cmath

# other import
import numpy as np


# local import
from ..core.error import SMRTError, smrt_warn
from ..core.globalconstants import C_SPEED
from .rayleigh import Rayleigh

#
# DMRT short range derives from Rayleigh because it has the same phase matrix
# only the scattering and absorption coefficient are different.
#

npol = 2


class DMRT_QCA_ShortRange(Rayleigh):

    """ DMRT electromagnetic model in the short range limit (grains AND aggregates are small) as implemented in DMRTML

        :param sensor: sensor instance
        :param layer: layer instance
        :dense_snow_correction: set how snow denser than half the ice density (ie. fractional volume larger than 0.5 is handled).
        "auto" means that snow is modeled as air bubble in ice instead of ice spheres in air.
        "bridging" should be developed in the future.
    """

    def __init__(self, sensor, layer, dense_snow_correction="auto"):

        # super().__init__()  # must not be called. Todo: write a generic RayleighBase object with phase function methods only

        if layer.frac_volume > 0.5 and dense_snow_correction == "auto":
            layer = layer.inverted_medium()

        f = layer.frac_volume
        if f > 0.5:
            smrt_warn("Using DMRT with fraction_volume > 0.5 is not recommended, unless for testing."
                      " See Picard et al. 2022 and references therein (doi: 10.5194/tc-16-3861-2022)"
                      " for a detailed description of the issue.")

        e0 = layer.permittivity(0, sensor.frequency)  # background permittivity
        es = layer.permittivity(1, sensor.frequency)  # scatterer permittivity

        lmda = C_SPEED / sensor.frequency

        if not hasattr(layer.microstructure, "stickiness") or not hasattr(layer.microstructure, "compute_t"):
            raise SMRTError("DMRT_QCA_ShortRange is only compatible with SHS microstructure model")

        radius = layer.microstructure.radius
        t = layer.microstructure.compute_t()

        y = (es - e0) / (es + 2 * e0)

        fy = f * y

        k0 = (2 * math.pi / lmda) * cmath.sqrt(e0).real
        Eeff = e0 + 3 * fy * e0 / (1 - fy) * (1 + 2j / 3 * (k0 * radius)**3 * y
                                              * (1 - f)**4 / ((1 - fy) * (1 + 2 * f - t * f * (1 - f))**2))
        Ks = 2 / (9 * f) * k0 * (k0 * radius)**3 * (
            np.abs(Eeff / e0 - 1)**2 * (1 - f)**4 / (1 + 2 * f - t * f * (1 - f))**2)  # TODO: to further double check

        beta = 2 * k0 * cmath.sqrt(Eeff).imag

        if Ks >= beta:
            smrt_warn("Grain diameter is too large for DMRT_QCACP_ShortRange resulting in single scattering albedo "
                      "larger than 1. It is recommended to decrease the size or used an alternative emmodel able to do "
                      "Mie calculations.")

        self._effective_permittivity = Eeff
        self.ks = Ks
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
