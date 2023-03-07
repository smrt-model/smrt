# coding: utf-8

"""Implement the empirical soil model presented in Wegmuller and Maetzler 1999. 
It is often used in microwave radiometry. It is not suitable for the active mode.

parameters: roughness_rms

"""

import numpy as np

# local import
from smrt.core.interface import Substrate
from smrt.core.fresnel import fresnel_reflection_matrix, fresnel_transmission_matrix


class SoilWegmuller(Substrate):

    args = ['roughness_rms']
    optional_args = {}

    def adjust(self, rh, rv, frequency, eps_1, mu1):
        # in place modification of rh and rv for the rough soil reflectivity model of Wegmüller & Mätzler (1999)

        #  Calculate ksigma = wavenumber*soilp%sigma(standard deviation of surface height)

        ksigma = 2 * np.pi * frequency * np.sqrt((1 / 2.9979e8)**2 * eps_1) * self.roughness_rms
        ksigma = ksigma.real

        #  Calculation of rh with ksigma
        rh *= np.exp(-ksigma**(np.sqrt(0.1 * mu1)))  # H pola

        # calculation of rv with rh (the model is valid for angle between 0-70°

        mask = mu1 < np.cos(60 * np.pi / 180)

        rv[~mask] = rh[~mask] * mu1[~mask]**0.655  # <-- * ou ** ??
        rv[mask] = rh[mask] * (0.635 - 0.0014 * (np.arccos(mu1[mask]) * 180 / np.pi - 60))

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
