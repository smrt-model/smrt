# coding: utf-8

"""Monodisperse sticky hard sphere model of the microstructure. This microstructure uses unified parameters as defined by 
G. Picard, H. Löwe, F. Domine, L. Arnaud, F. Larue, V. Favier, E. Le Meur, E. Lefebvre, J. Savarino, A. Royer, The snow microstructural control on microwave scattering, AGU Advances.

parameters: frac_volume, porod_length, polydispersity

"""

import numpy as np

from .unified_autocorrelation import UnifiedAutocorrelation


class UnifiedStickyHardSpheres(UnifiedAutocorrelation):
    """
    """

    def __init__(self, params):

        super().__init__(params)  # don't forget this line in our classes!

        # value of the correlation function at the origin
        self.corr_func_at_origin = self.frac_volume * (1.0 - self.frac_volume)

        self.radius = 3 / 4 * self.porod_length / (1 - self.frac_volume)

        K_32 = self.polydispersity**(-3 / 2)
        self.t = (1 + 2 * self.frac_volume - 3 / (8 * np.sqrt(2)) * K_32) / self.corr_func_at_origin

        # t must check the condition t * f (1-f) < 1 + 2*f  which seems to be always valid...

    def basic_check(self):
        """check consistency between the parameters"""
        pass

    # No analytical function exists
    # def autocorrelation_function(self, r):
    #
    def compute_stickiness(self):

        phi_2 = self.frac_volume
        return phi_2 / 12 * self.t - phi_2 / (1 - phi_2) + (1 + phi_2 / 2) / (self.t * (1 - phi_2)**2)

    def ft_autocorrelation_function(self, k):
        """Compute the 3D Fourier transform of the isotropic correlation
        function for sticky hard spheres in Percus--Yevick
        approximation for given magnitude k of the 3D wave vector
        (float).

        """

        # TODO LH:
        # * get solution for t directly from method compute_t
        # (this would include a check if the combination of stickiness
        # and volume fraction is admissible.
        # * check if k is positive (maybe not required since the function is even in k)

        d = 2 * self.radius
        phi_2 = self.frac_volume

        # scaling variable, Eq 32, LP2015
        X = np.atleast_1d(k) * d / 2.0

        # solution of the quadratic equation, Eq. 32, LP2015
        # sphere volume
        vd = 4.0 / 3 * np.pi * (d / 2.0)**3

        # number density
        n = phi_2 / vd

        # intersection volume, Eq. 27, LP2015
        sqrt_vint = np.empty_like(X)
        zerok = np.isclose(X, 0, atol=1e-03)
        nzerok = ~zerok
        sqrt_vint[nzerok] = vd * 3 * (np.sinc(X[nzerok] / np.pi) - np.cos(X[nzerok])) / X[nzerok]**2  # sqrt(intersection volume )* X²
        sqrt_vint[zerok] = vd

        # Ghislain says: the following quantities are already multiplied by the sqrt_vint_X2 to avoid singularity in 0.
        # this differs from the original equations where the vint is multiplied at the end.

        # auxiliary quantities defined in Tsang II Eq.8.4.19-8.4.22
        Psi_tsang_vol = np.sinc(X / np.pi) / sqrt_vint
        Phi_tsang_vol = 1.0 / vd  # Ghislain says: after simple math, vint sqrt_vint_X2 simplifies

        # auxiliary quantities Eq 31, LP2015
        A_tsang_vol = (phi_2 / (1 - phi_2) * ((1 - self.t * phi_2 + 3 * phi_2 / (1 - phi_2)) * Phi_tsang_vol +
                                              (3 - self.t * (1 - phi_2)) * Psi_tsang_vol) + np.cos(X) / sqrt_vint)
        B_tsang_vol = phi_2 / (1 - phi_2) * X * Phi_tsang_vol + np.sin(X) / sqrt_vint

        # structure factor Eq 31, LP2015
        S_tsang_vol = 1 / (A_tsang_vol**2 + B_tsang_vol**2)

        # FT correlation function, Eq. 25, LP2015
        Ctilde = n * S_tsang_vol

        # set limit value at k=0 manually, Eq. 33, LP2015
        # zerok = np.isclose(X, 0)
        # Ctilde[zerok] = (n * vd**2 / (phi_2 / (1-phi_2) * ((1 - t*phi_2 + 3 * phi_2 / (1 - phi_2)) + (3 - t * (1 - phi_2))) + 1)**2)
        Ctilde[zerok] = phi_2 * vd / (phi_2 / (1 - phi_2) * (
            (1 - self.t * phi_2 + 3 * phi_2 / (1 - phi_2)) + (3 - self.t * (1 - phi_2))) + 1)**2

        return Ctilde
