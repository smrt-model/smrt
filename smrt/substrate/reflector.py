# coding: utf-8

""" Implement a reflective boundary conditions with prescribed reflection coefficient in the specular direction.
The reflection is set to a value or a function of theta. Azimuthal symmetry is assumed (no dependence on phi).

The `specular_reflection` parameter can be a scalar, a function or a dictionary.

    - scalar: same reflection is use for all angles
    - function: the function must take a unique argument theta array (in radians) and return the reflection as an array of the same size as theta
    - dictionary: in this case, the keys must be 'H' and 'V' and the values are a scalar or a function and are interpreted as for the non-polarized case.

To make a reflector, it is recommended to use the helper function :py:func:`~smrt.substrate.reflector.make_reflector`.


Examples::

    # the full path import is required
    from smrt.substrate.reflector import make_reflector

    # return a perfect reflector (the temperature is useless in this specific case)
    ref = make_reflector(temperature=260, specular_reflection=1)

    # return a perfect absorber / black body.
    ref = make_reflector(temperature=260, specular_reflection=0)

    # Specify a frequency and polarization dictionary of reflectivity
    ref = make_reflector(specular_reflection={(21e9, 'H'): 0.5, (21e9, 'V'): 0.6, (36e9, 'H'): 0.7, (36e9, 'V'): 0.8})

.. note::

    the backscatter coefficient argument is not implemented/documented yet.

"""

import numpy as np

# local import
from smrt.core.interface import Substrate
from smrt import SMRTError
from smrt.core import lib
from smrt.core.lib import smrt_matrix


def make_reflector(temperature=None, specular_reflection=None):

    """ Construct a reflector or absorber instance.

    """

    # create the instance
    return Reflector(temperature=temperature, specular_reflection=specular_reflection)


class Reflector(Substrate):

    args = []
    optional_args = {'specular_reflection': None, 'backscatter_coefficient': None}

    def specular_reflection_matrix(self, frequency, eps_1, mu1, npol):

        if npol > 2:
            raise NotImplementedError("active model is not yet implemented, need modification for the third component")

        if self.backscatter_coefficient is not None:
            raise NotImplementedError("backscatter_coefficient to be implemented")

        if self.specular_reflection is None and self.backscatter_coefficient is None:
            self.specular_reflection = 1

        spec_refl_coeff = smrt_matrix.zeros((npol, len(mu1)))

        spec_refl_coeff[0] = self._get_refl(frequency, 'V', mu1)
        spec_refl_coeff[1] = self._get_refl(frequency, 'H', mu1)

        return spec_refl_coeff

    def emissivity_matrix(self, frequency, eps_1, mu1, npol):

        if self.specular_reflection is None and self.backscatter_coefficient is None:
            self.specular_reflection = 1

        if npol > 2:
            raise NotImplementedError("active model is not yet implemented, need modification for the third component")

        emissivity = smrt_matrix.zeros((npol, len(mu1)))

        emissivity[0] = 1 - self._get_refl(frequency, 'V', mu1)
        emissivity[1] = 1 - self._get_refl(frequency, 'H', mu1)

        return emissivity

    def _get_refl(self, frequency, polarization, mu1):

        specular_reflection = self.specular_reflection

        # try to get the frequency and/or the polarization if it is a dict
        if isinstance(specular_reflection, dict):
            for key in [(frequency, polarization), (polarization, frequency), frequency, polarization]:
                if key in specular_reflection:
                    specular_reflection = specular_reflection[key]
                    break

        if isinstance(specular_reflection, dict):
            raise SMRTError("The specular_reflection argument must be a scalar or a dict with the frequency and/or polarization as a key. If both, provide frequency and polarization as a tuple key")

        if callable(specular_reflection):  # we have a function, call it and see what we get
            user_refl = specular_reflection(np.arccos(mu1))
            if len(user_refl) == len(mu1):  # we have only one polarization
                return user_refl
            else:
                raise SMRTError("The length/shape or the specular_reflection function is incorrect")
        else:  # we have a scalar
            return np.full(len(mu1), specular_reflection)
