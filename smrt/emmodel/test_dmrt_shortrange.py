# coding: utf-8


import numpy as np

from smrt.emmodel.dmrt_qcacp_shortrange import DMRT_QCACP_ShortRange
from smrt.core.error import SMRTError
from smrt.core.globalconstants import  DENSITY_OF_ICE
from smrt.inputs.sensor_list import amsre
from smrt.inputs.make_medium import make_snow_layer
from smrt.emmodel import commontest
from smrt.permittivity.ice import ice_permittivity_maetzler06  # default ice permittivity model


# import the microstructure
from smrt.microstructure_model.sticky_hard_spheres import StickyHardSpheres
tolerance_pc = 0.01  # 1% tolerance


def setup_func_shs():
    # ### Make a snow layer
    shs_lay = make_snow_layer(layer_thickness=0.2, microstructure_model=StickyHardSpheres, density=250, temperature=265, radius=1e-3, stickiness=0.2)
    return shs_lay


def setup_func_em(testpack=None):
    if testpack is None:
        testpack = setup_func_shs()
    sensor = amsre('37V')
    emmodel = DMRT_QCACP_ShortRange(sensor, testpack)
    return emmodel


def setup_mu(stepsize):
    mu_pos = np.arange(1.0, 0., - stepsize)
    mu_neg = - mu_pos
    mu = np.concatenate((mu_pos, mu_neg))
    mu = np.array(mu)
    return mu


def test_energy_conservation():

    em = setup_func_em()
    commontest.test_energy_conservation(em, tolerance_pc)



def setup_func_dense_shs(density, inverse=False):
    # ### Make a snow layer

    if inverse:
        density = DENSITY_OF_ICE - density
        eps_scatterers = 1
        eps_background = ice_permittivity_maetzler06

        shs_lay = make_snow_layer(layer_thickness=0.2, microstructure_model=StickyHardSpheres,
                                density=density, temperature=265, radius=0.2e-3, stickiness=0.2,
                                ice_permittivity_model=eps_scatterers,
                                background_permittivity_model=eps_background)
    else:
        shs_lay = make_snow_layer(layer_thickness=0.2, microstructure_model=StickyHardSpheres,
                                density=density, temperature=265, radius=0.2e-3, stickiness=0.2)

    return shs_lay


def test_invert_dmrt():

    em = setup_func_em(setup_func_dense_shs(250))
    em_inv = setup_func_em(setup_func_dense_shs(250, inverse=True))
    assert (abs(em.ks - em_inv.ks) < 1e-4)

    em = setup_func_em(setup_func_dense_shs(700))
    em_inv = setup_func_em(setup_func_dense_shs(700, inverse=True))
    assert (abs(em.ks - em_inv.ks) < 1e-4)