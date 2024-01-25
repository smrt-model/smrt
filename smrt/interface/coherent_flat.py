

"""
Implement the coherent flat pseudo-interface, as in MEMLS. This interface is obtained by collapsing one layer and two interfaces into a single interface. Scattering in the layer is neglected.


"""

import numpy as np
from smrt.core.globalconstants import C_SPEED
from smrt.core.lib import smrt_matrix, abs2
from smrt.core.fresnel import fresnel_coefficients
from smrt.core.error import SMRTError


def process_coherent_layers(snowpack, emmodel_list, sensor):

    effective_permittivity = [em.effective_permittivity() for em in emmodel_list]
    phase = [sensor.wavenumber * np.sqrt(eps_eff).real * lay.thickness for lay, eps_eff in zip(snowpack.layers, effective_permittivity)]

    coherent_layers = np.array(phase) < 3 * np.pi / 4

    if not np.any(coherent_layers):
        return snowpack, emmodel_list

    snowpack = snowpack.copy()

    if coherent_layers[-1]:
        raise SMRTError("The last layer is coherent, this is not supported")

    print("process_coherent_layers (in dev, use for testing only) # coherent layers:", np.sum(coherent_layers))

    for l in np.flatnonzero(coherent_layers[:-1])[::-1]:  # reverse the processing to safely delete the snowpack layer and interface
        print("coherent layer:", l)
        if coherent_layers[l - 1]:
            raise SMRTError("Two sucessive layers are coherent, this is not yet supported")
        # create a coherent interface
        coherent_interface = CoherentFlat(snowpack.interfaces[l:l + 2], snowpack.layers[l], effective_permittivity[l])
        # set the next interface to coherent
        snowpack.interfaces[l + 1] = coherent_interface
        # delete the layer to be deleted
        snowpack.delete(l)  # delete layer and interface l
        emmodel_list.pop(l)

    return snowpack, emmodel_list


class CoherentFlat(object):
    """A flat surface. The reflection is in the specular direction and the coefficient is calculated with the Fresnel coefficients

"""
    args = []
    optional_args = {}

    def __init__(self, interfaces, layer, permittivity):

        super().__init__()

        self.interfaces = interfaces  # (interface_above, interface_below)
        self.layer = layer
        self.permittivity = permittivity

    def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """compute the reflection coefficients for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium.
        :param mu1: array of cosine of incident angles.
        :param npol: number of polarization.

        :return: the reflection matrix
"""

        R01_v, R01_h, R1t_v, R1t_h, exp_kd, exp_2kd, mu_t = self._prepare_computation(frequency, eps_1, eps_2, mu1)

        R_v = (R01_v + R1t_v * exp_2kd) / (1 + R01_v * R1t_v * exp_2kd)
        R_h = (R01_h + R1t_h * exp_2kd) / (1 + R01_h * R1t_h * exp_2kd)

        reflection_coefficients = smrt_matrix.ones((npol, len(mu1)))

        reflection_coefficients[0] = abs2(R_v)
        reflection_coefficients[1] = abs2(R_h)

        if npol >= 3:
            reflection_coefficients[2] = (R_v * np.conj(R_h)).real   # TsangI  Eq 7.2.93

        return reflection_coefficients

    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        return smrt_matrix(0)

    def coherent_transmission_matrix(self, frequency, eps_1, eps_2, mu1, npol):
        """compute the transmission coefficients for the azimuthal mode m
           and for an array of incidence angles (given by their cosine)
           in medium 1. Medium 2 is where the beam is transmitted.

        :param eps_1: permittivity of the medium where the incident beam is propagating.
        :param eps_2: permittivity of the other medium.
        :param mu1: array of cosine of incident angles.
        :param npol: number of polarization.

        :return: the transmission matrix
"""

        R01_v, R01_h, R1t_v, R1t_h, exp_kd, exp_2kd, mu_t = self._prepare_computation(frequency, eps_1, eps_2, mu1)

        T_v = (1 + R01_v) * (1 + R1t_v) * exp_kd / (1 + R01_v * R1t_v * exp_2kd)  # see TsnagI 5.2.10-12
        T_h = (1 + R01_h) * (1 + R1t_h) * exp_kd / (1 + R01_h * R1t_h * exp_2kd)

        transmission_coefficients = smrt_matrix.ones((npol, len(mu1)))

        nt = np.sqrt(eps_2 / eps_1).real
        transmission_coefficients[0] = abs2(T_v) * mu_t / mu1 / nt  # for the coef see TsangIII 2.1.140b
        transmission_coefficients[1] = abs2(T_h) * mu_t / mu1 * nt  # for the coef see TsangIII 2.1.140a

        if npol >= 3:
            # this part is to be confirmed.
            R_v = (R01_v + R1t_v * exp_2kd) / (1 + R01_v * R1t_v * exp_2kd)
            R_h = (R01_h + R1t_h * exp_2kd) / (1 + R01_h * R1t_h * exp_2kd)

            transmission_coefficients[2] = mu_t / mu1 * ((1 + R_v) * np.conj(1 + R_h)).real  # TsangI  Eq 7.2.95

        return transmission_coefficients

    def _prepare_computation(self, frequency, eps_1, eps_2, mu1):

        # convert to Tsang's notation. See TsangI, pages 207, Eq. 5.2.14
        eps_0 = eps_1
        eps_1 = self.permittivity
        eps_t = eps_2

        mu_0 = mu1
        R01_v, R01_h, mu_1 = fresnel_coefficients(eps_0, eps_1, mu_0)
        R1t_v, R1t_h, mu_t = fresnel_coefficients(eps_1, eps_t, np.maximum(mu_1, 1e-4))

        k_1 = 2 * np.pi / C_SPEED * frequency * np.sqrt(eps_1)

        phase = k_1 * mu_1 * self.layer.thickness
        assert np.all(phase.imag >= 0)

        incoherent = phase.real > 3 * np.pi / 4  # we consider coherency up to 3 pi / 2 like in MEMLS

        phase[incoherent].real = 0

        exp_kd = np.exp(1j * phase)
        exp_2kd = np.exp(2j * phase)

        return R01_v, R01_h, R1t_v, R1t_h, exp_kd, exp_2kd, mu_t

    def diffuse_transmission_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
        return smrt_matrix(0)
