# coding: utf-8

import pytest

import numpy as np

from smrt.emmodel.iba_original import IBA_original
from smrt.emmodel.rayleigh import Rayleigh
from smrt.core.error import SMRTError
from smrt.core.sensor import active
from smrt.inputs.sensor_list import amsre
from smrt import make_snow_layer
from smrt.emmodel import commontest

# import the microstructure
from smrt.microstructure_model.exponential import Exponential
from smrt.microstructure_model.independent_sphere import IndependentSphere
from smrt.microstructure_model.sticky_hard_spheres import StickyHardSpheres

tolerance = 1e-7
tolerance_pc = 0.05  # 5% error is allowable for differences from MEMLS values. Tests pass at 2%. Some fail at 1%.


def setup_func_sp():
    # Could import iba_example, but hard code here in case iba_example changes
    # ### Make a snow layer
    exp_lay = make_snow_layer(layer_thickness=0.2, microstructure_model=Exponential, density=250, temperature=265, corr_length=5e-4)
    return exp_lay


def setup_func_indep(radius=5e-4):
    # ### Make a snow layer
    indep_lay = make_snow_layer(layer_thickness=0.2, microstructure_model=IndependentSphere, density=250, temperature=265, radius=radius)
    return indep_lay


def setup_func_shs():
    # ### Make a snow layer
    shs_lay = make_snow_layer(layer_thickness=0.2, microstructure_model=StickyHardSpheres, density=250, temperature=265, radius=5e-4, stickiness=0.2)
    return shs_lay


def setup_func_pc(pc):
    # ### Make a snow layer
    exp_lay = make_snow_layer(layer_thickness=0.1, microstructure_model=Exponential, density=300, temperature=265, corr_length=pc)
    return exp_lay


def setup_func_em(testpack=None):
    if testpack is None:
        testpack = setup_func_sp()
    sensor = amsre('37V')
    emmodel = IBA_original(sensor, testpack)
    return emmodel


def setup_func_active(testpack=None):
    if testpack is None:
        testpack = setup_func_sp()
    scatt = active(frequency=10e9, theta_inc=50)
    emmodel = IBA_original(scatt, testpack)
    return emmodel


def setup_func_rayleigh():
    testpack = setup_func_indep(radius=1e-4)
    sensor = amsre('10V')
    emmodel_iba = IBA_original(sensor, testpack)
    emmodel_ray = Rayleigh(sensor, testpack)
    return emmodel_iba, emmodel_ray


def setup_mu(stepsize, bypass_exception=None):
    mu_pos = np.arange(1.0, 0., - stepsize)
    if bypass_exception:
        # exclude mu = 1
        mu_pos = mu_pos[1:]
    mu_neg = - mu_pos
    mu = np.concatenate((mu_pos, mu_neg))
    mu = np.array(mu)
    return mu


# Tests to compare with MEMLS IBA, graintype = 2 (small spheres) outputs


def test_ks_pc_is_0p3_mm():
    testpack = setup_func_pc(0.3e-3)
    em = setup_func_em(testpack)
    # Allow 5% error
    memls_ks = 4.13718676e+00
    assert abs(em.ks - memls_ks) < tolerance_pc * em.ks


def test_ks_pc_is_0p25_mm():
    testpack = setup_func_pc(0.25e-3)
    em = setup_func_em(testpack)
    # Allow 5% error
    memls_ks = 2.58158887e+00
    # eq_(em.ks, memls_ks)
    assert abs(em.ks - memls_ks) < tolerance_pc * em.ks


def test_ks_pc_is_0p2_mm():
    testpack = setup_func_pc(0.2e-3)
    em = setup_func_em(testpack)
    # Allow 5% error
    memls_ks = 1.41304849e+00
    assert abs(em.ks - memls_ks) < tolerance_pc * em.ks


def test_ks_pc_is_0p15_mm():
    testpack = setup_func_pc(0.15e-3)
    em = setup_func_em(testpack)
    # Allow 5% error
    memls_ks = 6.30218291e-01
    assert abs(em.ks - memls_ks) < tolerance_pc * em.ks


def test_ks_pc_is_0p1_mm():
    testpack = setup_func_pc(0.1e-3)
    em = setup_func_em(testpack)
    # Allow 5% error
    memls_ks = 1.94727497e-01
    assert abs(em.ks - memls_ks) < tolerance_pc * em.ks


def test_ks_pc_is_0p2_mm():
    testpack = setup_func_pc(0.05e-3)
    em = setup_func_em(testpack)
    # Allow 5% error
    memls_ks = 2.49851702e-02
    assert abs(em.ks - memls_ks) < tolerance_pc * em.ks


def test_energy_conservation_exp():
    em = setup_func_em()
    commontest.test_energy_conservation(em, tolerance_pc)


def test_energy_conservation_indep():
    indep_pack = setup_func_indep()
    em = setup_func_em(testpack=indep_pack)
    commontest.test_energy_conservation(em, tolerance_pc)


def test_energy_conservation_shs():
    shs_pack = setup_func_shs()
    em = setup_func_em(testpack=shs_pack)
    commontest.test_energy_conservation(em, tolerance_pc)


def test_npol_passive_is_2():
    em = setup_func_em()
    assert em.npol == 2


def test_npol_active_is_3():
    em = setup_func_active()
    assert em.npol == 3


def test_energy_conservation_exp_active():
    em = setup_func_active()
    commontest.test_energy_conservation(em, tolerance_pc, npol=2)


def test_energy_conservation_indep_active():
    indep_pack = setup_func_indep()
    em = setup_func_active(testpack=indep_pack)
    commontest.test_energy_conservation(em, tolerance_pc, npol=2)


def test_energy_conservation_shs_active():
    shs_pack = setup_func_shs()
    em = setup_func_active(testpack=shs_pack)
    commontest.test_energy_conservation(em, tolerance_pc, npol=2)


# def test_energy_conservation_shs_active_but_npol_is_2():
#     shs_pack = setup_func_shs()
#     em = setup_func_active(testpack=shs_pack)
#     commontest.test_energy_conservation(em, tolerance_pc, npol=2)


def test_iba_vs_rayleigh_passive_m0():
    em_iba, em_ray = setup_func_rayleigh()
    mu = setup_mu(1. / 64)
    assert (abs(em_iba.ft_even_phase(mu, mu, 0, npol=2) / em_iba.ks
                - em_ray.ft_even_phase(mu, mu, 0, npol=2) / em_ray.ks) < tolerance_pc).all()


def test_iba_vs_rayleigh_active_m0():
    # Have to set npol = 2 for m=0 mode in active otherwise rayleigh will produce 3x3 matrix
    em_iba, em_ray = setup_func_rayleigh()
    mu = setup_mu(1. / 64, bypass_exception=True)
    assert (abs(em_iba.ft_even_phase(mu, mu, 0, npol=2) / em_iba.ks
                - em_ray.ft_even_phase(mu, mu, 0, npol=2) / em_ray.ks) < tolerance_pc).all()


def test_iba_vs_rayleigh_active_m1():
    em_iba, em_ray = setup_func_rayleigh()
    mu = setup_mu(1. / 64, bypass_exception=True)
    # Clear cache
    em_iba.cached_mu = None
    assert (abs(em_iba.ft_even_phase(mu, mu, 1, npol=3)[:, :, 1] / em_iba.ks
                - em_ray.ft_even_phase(mu, mu, 1, npol=3)[:, :, 1] / em_ray.ks) < tolerance_pc).all()


def test_iba_vs_rayleigh_active_m2():
    em_iba, em_ray = setup_func_rayleigh()
    mu = setup_mu(1. / 64, bypass_exception=True)

    def check(i, j):
        print(em_iba.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_iba.ks,
            abs(em_ray.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_ray.ks))
        assert abs((abs(em_iba.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_iba.ks)
                    - abs(em_ray.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_ray.ks)) < tolerance_pc).all()

        assert (abs(em_iba.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_iba.ks
                    - em_ray.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_ray.ks) < tolerance_pc).all()

    check(0, 0)
    check(0, 1)
    check(0, 2)
    check(1, 0)
    check(1, 1)
    check(1, 2)
    check(2, 0)
    check(2, 1)
    check(2, 2)


def test_iba_raise_exception_mu_is_1():
    shs_pack=setup_func_shs()
    em=setup_func_active(testpack=shs_pack)
    bad_mu=np.array([0.2, 1])
    with pytest.raises(SMRTError):
        em.ft_even_phase(bad_mu, bad_mu, 2, npol=3)[:, :, 2]
