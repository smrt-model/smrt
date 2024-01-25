# coding: utf-8

"""Implement the rough boundary reflectivity presented in Choudhury et al. (1979). It is not suitable for the active mode.

Applicable for ksigma<<1

parameters: roughness_rms

"""

import numpy as np

# local import
from smrt.core.interface import Substrate
from smrt.core.fresnel import fresnel_reflection_matrix, fresnel_transmission_matrix


class ChoudhuryReflectivity(Substrate):

    args = ['roughness_rms']
    optional_args = {}

    def adjust(self, rh, rv, frequency, eps_1, mu1):
        # in place modification of rh and rv for the rough reflectivity model of Choudhury et al. (1979)

        #  Calculate ksigma = wavenumber*soilp%sigma(standard deviation of surface height)

        ksigma = 2 * np.pi * frequency * np.sqrt((1 / 2.9979e8)**2 * eps_1) * self.roughness_rms
        ksigma = ksigma.real

        # Raise warning if outside validity
        if ksigma > 0.1:
            raise Warning("Reflectivity may be outside validity range. ksigma should be << 1")

        #  Calculation of rh with ksigma
        rh *= np.exp(-4 * ksigma**2 * mu1**2)  # H pola
        rv *= np.exp(-4 * ksigma**2 * mu1**2)  # V pola

    def specular_reflection_matrix(self, frequency, eps_1, mu1, npol):

        eps_2 = self.permittivity(frequency)

        reflection_coefficients = fresnel_reflection_matrix(eps_1, eps_2, mu1, npol)

        self.adjust(reflection_coefficients[1], reflection_coefficients[0], frequency, eps_1, mu1)

        if npol >= 3:
            # don't modify the third compoment... this is an approximation, as the third component should be affected by the roughness...
            # don't use this model for active mode
            pass
        if npol == 4:
            raise NotImplementedError("to be implemented, the matrix is not diagonal anymore")

        return reflection_coefficients

    def emissivity_matrix(self, frequency, eps_1, mu1, npol):

        # this function is a bit complex because we have to change first and second component but not the third one.
        # this is an approximation, as the third component should be affected by the roughness...

        eps_2 = self.permittivity(frequency)

        transmission_coefficients = fresnel_transmission_matrix(eps_1, eps_2, mu1, npol)

        rh = 1 - transmission_coefficients[1]
        rv = 1 - transmission_coefficients[0]

        self.adjust(rh, rv, frequency, eps_1, mu1)

        transmission_coefficients[1] = 1 - rh  # back to transmission coefficients
        transmission_coefficients[0] = 1 - rv

        if npol >= 3:
            # don't modify the third compoment... don't know what to do with it !
            pass
        if npol == 4:
            raise NotImplementedError("to be implemented, the matrix is not diagonal anymore")

        return transmission_coefficients
