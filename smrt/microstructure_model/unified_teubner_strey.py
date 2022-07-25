# coding: utf-8

"""Extended Teubner Strey model as described by Ruland 2010. This microstructure uses unified parameters as defined by 
G. Picard, H. Löwe, F. Domine, L. Arnaud, F. Larue, V. Favier, E. Le Meur, E. Lefebvre, J. Savarino, A. Royer, The snow microstructural control on microwave scattering, AGU Advances.

parameters: frac_volume, porod_length, polydispersity

"""

import numpy as np

from .unified_autocorrelation import UnifiedAutocorrelation


class UnifiedTeubnerStrey(UnifiedAutocorrelation):

    def __init__(self, params):

        super().__init__(params)  # don't forget this line in our classes!

        # value of the correlation function at the origin
        self.corr_func_at_origin = self.frac_volume * (1.0 - self.frac_volume)

        K32 = self.polydispersity ** (3 / 2)

        if self.polydispersity >= 1:
            # Case 1 according to Ruland 2010
            b = self.porod_length * K32
            delta = np.sqrt(1 - 1 / K32)
            self.zeta1 = b * (1 - delta)
            self.zeta2 = b * (1 + delta)
        else:
            # Case 2 according to Ruland 2010
            self.zeta1 = self.porod_length
            self.zeta2 = self.porod_length * np.sqrt(1 / (1 / K32 - 1))

    def basic_check(self):
        """check consistency between the parameters"""
        pass

    def compute_ssa(self):
        """Compute the ssa for a sphere"""
        # implement here the calculation of the SSA from the microstructure parameters.
        raise NotImplementedError("to be implemented")

    def autocorrelation_function(self, r):
        """compute the real space autocorrelation function for the Teubner Strey model

        """

        if self.polydispersity >= 1:
            inv_harmonic_mean = 1 / self.zeta1 - 1 / self.zeta2
            denom = r * inv_harmonic_mean
            expc = np.where(denom > 1e-15, (np.exp(-r / self.zeta2) - np.exp(-r / self.zeta1)) / denom, 1)
            acf = self.corr_func_at_origin * expc

        else:
            acf = self.corr_func_at_origin * np.exp(-r / self.zeta1) * np.sinc(r / self.zeta2 / np.pi)

        return acf

    def ft_autocorrelation_function(self, k):
        """Compute the 3D Fourier transform of the isotropic correlation
        function for Teubner Strey for given magnitude k of the 3D wave vector
        (float).

        """

        if self.polydispersity >= 1:
            ft_acf_normalized = 4 * np.pi * self.zeta1 * self.zeta2 * (self.zeta1 + self.zeta2) / \
                ((1 + (self.zeta1 * k)**2) * (1 + (self.zeta2 * k)**2))
        else:
            x1 = k * self.zeta1
            r12 = self.zeta1 / self.zeta2
            ft_acf_normalized = 8 * np.pi * self.zeta1**3 / ((1 + (x1 - r12)**2) * (1 + (x1 + r12)**2))

        return self.corr_func_at_origin * ft_acf_normalized
