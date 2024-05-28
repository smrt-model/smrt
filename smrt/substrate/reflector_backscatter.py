# coding: utf-8

""" Implement a reflective boundary conditions with prescribed reflection coefficient in the specular direction, and backscatter coefficient.
The reflection is set to a value or a function of theta. Azimuthal symmetry is assumed (no dependence on phi). 

The `specular_reflection` parameter can be a scalar, a function or a dictionary.

    - scalar: same reflection is use for all angles
    - function: the function must take a unique argument theta array (in radians) and return the reflection as an array of the same size as theta
    - dictionary: in this case, the keys must be 'H' and 'V' and the values are a scalar or a function and are interpreted as for the non-polarized case.

The `backscattering_coefficient` is a dictionary with VV nad HH keys. It is not possible to set HV and VH.
Note also that modeling substrate with prescribed backscatter value with the DORT solver is an approximate trick, and
the result is only approximatly the prescribed value even for a transparent snowpack.

To make a reflector, it is recommended to use the helper function :py:func:`~smrt.substrate.reflector.make_reflector`.


Examples::

    # the full path import is required
    from smrt.substrate.reflector import make_reflector

    # return a perfect reflector (the temperature is useless in this specific case)
    ref = make_reflector(temperature=260, specular_reflection=1)

    # return a perfect absorber / black body.
    ref = make_reflector(temperature=260, specular_reflection=0)

.. note::

    the backscatter coefficient argument is not implemented/documented yet.

"""

import numpy as np

# local import
from smrt.core.interface import Substrate
from smrt import SMRTError
from smrt.core.lib import smrt_matrix


def make_reflector(temperature=None, specular_reflection=None, backscattering_coefficient=None):

    """ Construct a reflector or absorber instance.

    """

    # create the instance
    return ReflectorBackscatter(temperature=temperature, specular_reflection=specular_reflection,
                     backscattering_coefficient=backscattering_coefficient)


class ReflectorBackscatter(Substrate):

    args = []
    optional_args = {'specular_reflection': None, 'backscattering_coefficient': None}

    def specular_reflection_matrix(self, frequency, eps_1, mu1, npol):

        if npol > 2 and not hasattr(ReflectorBackscatter, "stop_pol2_warning"):
            print("active model is not yet fully implemented, need modification for the third component")  # !!!
            ReflectorBackscatter.stop_pol2_warning = True

        if self.specular_reflection is None and self.backscattering_coefficient is None:
            self.specular_reflection = 1

        spec_refl_coeff = smrt_matrix.zeros((npol, len(mu1)))
        if isinstance(self.specular_reflection, dict):  # we have a dictionary with polarization
            spec_refl_coeff[0] = self._get_refl(self.specular_reflection['V'], mu1)
            spec_refl_coeff[1] = self._get_refl(self.specular_reflection['H'], mu1)
        else:  # we have a scalar, both polarization are the same
            spec_refl_coeff[0] = spec_refl_coeff[1] = self._get_refl(self.specular_reflection, mu1)

        return spec_refl_coeff

    def ft_even_diffuse_reflection_matrix(self, frequency, eps_1, mu_s, mu_i, m_max, npol):

        assert mu_s is mu_i

        if isinstance(self.backscattering_coefficient, dict):  # we have a dictionary with polarization
            diffuse_refl_coeff = smrt_matrix.zeros((npol, m_max + 1, len(mu_i)))

            for m in range(m_max + 1):
                if m == 0:
                    coef = 0.5
                elif (m % 2) == 1:
                    coef = -1.0
                else:
                    coef = 1.0
                coef /= 4 * np.pi * mu_i    # SMRT requires scattering coefficient / 4 * pi

                coef /= m_max + 0.5  # ad hoc normalization to get the right backscatter. This is a trick to deal with the dirac.

                diffuse_refl_coeff[0, m, :] += coef * self._get_refl(self.backscattering_coefficient['VV'], mu_i)
                diffuse_refl_coeff[1, m, :] += coef * self._get_refl(self.backscattering_coefficient['HH'], mu_i)

        elif self.backscattering_coefficient is not None:
            raise SMRTError("backscattering_coefficient must be a dictionary with keys VV and HH")
        else:
            diffuse_refl_coeff = smrt_matrix(0)

        return diffuse_refl_coeff

    def emissivity_matrix(self, frequency, eps_1, mu1, npol):

        if self.specular_reflection is None and self.backscattering_coefficient is None:
            self.specular_reflection = 1

        if npol > 2 and not hasattr(self, "stop_pol2_warning"):
            print("active model is not yet fully implemented, need modification for the third component")  # !!!
            self.stop_pol2_warning = True

        emissivity = smrt_matrix.zeros((npol, len(mu1)))
        if isinstance(self.specular_reflection, dict):  # we have a dictionary with polarization
            emissivity[0] = 1 - self._get_refl(self.specular_reflection['V'], mu1)
            emissivity[1] = 1 - self._get_refl(self.specular_reflection['H'], mu1)
        else:  # we have a scalar, both polarization are the same
            emissivity[0] = emissivity[1] = 1 - self._get_refl(self.specular_reflection, mu1)

        return emissivity

    def _get_refl(self, specular_reflection, mu1):
        if callable(specular_reflection):  # we have a function, call it and see what we get
            user_refl = specular_reflection(np.arccos(mu1))
            if len(user_refl) == len(mu1):  # we have only one polarization
                return user_refl
            else:
                raise SMRTError("The length/shape or the specular_reflection function is incorrect")
        else:  # we have a scalar
            return np.full(len(mu1), specular_reflection, dtype=np.float64)
