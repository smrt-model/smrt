# coding: utf-8

"""Compute scattering from Improved Born Approximation theory. This model allows for different
microstructural models provided that the Fourier transform of the correlation function
may be performed. All properties relate to a single layer. The absorption is calculated with the original formula in Mätzler 1998

"""

# Stdlib import

# other import
from .iba import IBA


class IBA_original(IBA):

    """
    Original Improved Born Approximation electromagnetic model class.

    As with all electromagnetic modules, this class is used to create an electromagnetic
    object that holds information about the effective permittivity, extinction coefficient and
    phase function for a particular snow layer. Due to the frequency dependence, information
    about the sensor is required. Passive and active sensors also have different requirements on
    the size of the phase matrix as redundant information is not calculated for the
    passive case.

    :param sensor: object containing sensor characteristics
    :param layer: object containing snow layer characteristics (single layer)

    """

    def compute_ka(self):
        """ IBA absorption coefficient calculated from the low-loss assumption of a general lossy medium.

        Calculates ka from wavenumber in free space (determined from sensor), and effective permittivity
        of the medium (snow layer property)

        :returns: absorption coefficient [m :sup:`-1`]

        .. note::

            This may not be suitable for high density material

        """

        # equation from Matzler 1998 (original IBA98 paper) and Matzler and Wiesmann 1999
        return self.k0 * self.frac_volume * self.eps.imag * abs(self.mean_sq_field_ratio())
