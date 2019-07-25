# coding: utf-8

"""Surface with a backscatter of 4pi.

"""

import numpy as np

# local import
from smrt.core.interface import Interface
from smrt.core.lib import smrt_matrix, len_atleast_1d

class RadarCalibrationSphere(Interface):

    args = []
    optional_args = {}

    def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):

        return smrt_matrix(0)

    def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):

        m = smrt_matrix.zeros((npol, len_atleast_1d(dphi), len_atleast_1d(mu_i)))
        m[0:2, :, :] = 1.

        return m

    def ft_even_diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol):
        
        m = smrt_matrix.zeros((npol, m_max + 1, len_atleast_1d(mu_s)))

        # only mode 0, pola 0 and 1, are non-null
        m[0:2, 0, :] = 1.

        return m

    def coherent_transmission_matrix(self, frequency, eps_1, eps_2, mu1, npol):

        return smrt_matrix(0)