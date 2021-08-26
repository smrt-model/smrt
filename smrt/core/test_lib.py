# coding: utf-8

import numpy as np

from smrt.inputs.make_medium import make_snow_layer
from smrt.inputs.sensor_list import amsre
from smrt.core.lib import generic_ft_even_matrix
from smrt.emmodel.rayleigh import Rayleigh

from smrt.microstructure_model.independent_sphere import IndependentSphere
tolerance_pc = 0.01  # 5% tolerance


def setup_func_sp():
    # ### Make a snow layer
    shs_lay = make_snow_layer(layer_thickness=0.2, microstructure_model=IndependentSphere, density=250, temperature=265, radius=5e-4)
    return shs_lay


def setup_func_em(testpack=None):
    if testpack is None:
        testpack = setup_func_sp()
    sensor = amsre('37V')
    emmodel = Rayleigh(sensor, testpack)
    return emmodel


def test_generic_ft_even_matrix():

    em = setup_func_em()

    mu = np.arange(0.1, 1, 0.4)
    npol = 3

    def phase_function(dphi):
        return em.phase(mu, mu, dphi, npol=npol)

    ft_even_p = generic_ft_even_matrix(phase_function, m_max=2)

    ft_even_p2 = em.ft_even_phase(mu, mu, m_max=2, npol=npol)

    for m in [0, 1, 2]:
        print("mode=", m)
        assert np.allclose(ft_even_p[:, :, m, :, :], ft_even_p2[:, :, m, :, :])
