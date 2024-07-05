# coding: utf-8

"""Compute scattering with the Strong-Contrast Expansion (SCE) from Rechtsman and Torquato, 2008 adapted by Ghislain Picard (unpublished at time of writting).
This SCE is the local version valid for quasi-static frequency (i.e. low frequency or small scatterers). A non-local version has been devised
recently 

"""

# Stdlib import

# other import
import numpy as np
import scipy.integrate

# local import
from smrt.permittivity.generic_mixing_formula import maxwell_garnett_for_spheres
from .sce_common import SCEBase

#
# For developers: all emmodel must implement the `effective_permittivity`, `ke` and `phase` functions with the same arguments as here
# initialisation and precomputation can be done in the prepare method that is called only once for each layer whereas
# phase, ke and effective_permittivity can be called several times.
#


class SCER08(SCEBase):

    """
        To be documented
    """

    def __init__(self, sensor, layer):

        super().__init__(sensor, layer)

        self.A2 = self.compute_A2(self.k1, self.microstructure)
        self._ke, self.ks = self.compute_ke_ks()

        self._effective_permittivity = self.effective_permittivity()

        self.ka = self.compute_ka()

        # another way:
        # self.eps, self.e0 = self.eps.real, self.e0.real
        # self.ks = self.compute_ke()
        # restore back the complex values of eps and e0

    def compute_A2(self, Q, microstructure):
        """ Compute A2 using equation 26

        """

        # compute the real part
        p = 12  # number of samples. This should be adaptative depending on the size/wavelength

        # grid resolution, fraction of the unique characteristic scale we have
        maxr = 2**(p // 2) * microstructure.inv_slope_at_origin
        n = 2**p
        r = np.linspace(0, maxr, n + 1)
        y = r * microstructure.autocorrelation_function(r)

        integrale1 = scipy.integrate.romb(y, maxr / n)

        A2 = 2 * Q**2 * (integrale1 + 1j * Q * float(microstructure.ft_autocorrelation_function(0)) / (4 * np.pi))
        return A2

    def compute_ke(self):

        # equation Eq 29 in Rechtsman and Torquato 2008 is equivalent to Maxwell Garnet with an adjusted fractional volume
        # (it can be complex)
        adjusted_fractional = self.frac_volume / (1 - self.A2 / self.frac_volume * (self.eps - self.e0) / (self.eps + 2 * self.e0))

        Eeff = maxwell_garnett_for_spheres(adjusted_fractional, self.e0, self.eps)
        Eeff0 = maxwell_garnett_for_spheres(self.frac_volume, self.e0, self.eps)

        ke = 2 * self.k0 * np.sqrt(Eeff).imag

        return ke, ke - 2 * self.k0 * np.sqrt(Eeff0).imag

    def effective_permittivity(self):
        """ Calculation of complex effective permittivity of the medium,
        which is given by Maxwell Garnet in the case of Rechtsman and Torquato, 2008

        :returns: effective_permittivity: complex effective permittivity of the medium

        """

        return maxwell_garnett_for_spheres(self.frac_volume, self.e0, self.eps)
