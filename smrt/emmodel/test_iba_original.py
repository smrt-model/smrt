# coding: utf-8

import numpy as np
import pytest

from smrt import make_snow_layer
from smrt.core.error import SMRTError
from smrt.core.sensor import active
from smrt.emmodel import commontest
from smrt.emmodel.iba_original import IBA_original
from smrt.emmodel.rayleigh import Rayleigh
from smrt.inputs.sensor_list import amsre

# import the microstructure
from smrt.microstructure_model.exponential import Exponential
from smrt.microstructure_model.independent_sphere import IndependentSphere
from smrt.microstructure_model.sticky_hard_spheres import StickyHardSpheres

tolerance = 1e-7
tolerance_pc = 0.05  # 5% error is allowable for differences from MEMLS values. Tests pass at 2%. Some fail at 1%.


def setup_func_sp():
    # Could import iba_example, but hard code here in case iba_example changes
    # ### Make a snow layer
    exp_lay = make_snow_layer(
        layer_thickness=0.2,
        microstructure_model=Exponential,
        density=250,
        temperature=265,
        corr_length=5e-4,
    )
    return exp_lay


def setup_func_indep(radius=5e-4):
    # ### Make a snow layer
    indep_lay = make_snow_layer(
        layer_thickness=0.2,
        microstructure_model=IndependentSphere,
        density=250,
        temperature=265,
        radius=radius,
    )
    return indep_lay

@pytest.fixture
def setup_func_shs():
    # ### Make a snow layer
    shs_lay = make_snow_layer(
        layer_thickness=0.2,
        microstructure_model=StickyHardSpheres,
        density=250,
        temperature=265,
        radius=5e-4,
        stickiness=0.2,
    )
    return shs_lay


@pytest.fixture
def setup_func_pc(request):
    ### Make a snow layer 
    # request.param will be set by the test parameterization
    pc = request.param
    exp_lay = make_snow_layer(
        layer_thickness=0.1,
        microstructure_model=Exponential,
        density=300,
        temperature=265,
        corr_length=pc,
    )
    return exp_lay


def setup_func_em(testpack=None):
    if testpack is None:
        testpack = setup_func_sp()
    sensor = amsre("37V")
    emmodel = IBA_original(sensor, testpack)
    return emmodel


def setup_func_active(testpack=None):
    if testpack is None:
        testpack = setup_func_sp()
    scatt = active(frequency=10e9, theta_inc=50)
    emmodel = IBA_original(scatt, testpack)
    return emmodel

@pytest.fixture
def setup_func_rayleigh():
    testpack = setup_func_indep(radius=1e-4)
    sensor = amsre("10V")
    emmodel_iba = IBA_original(sensor, testpack)
    emmodel_ray = Rayleigh(sensor, testpack)
    return emmodel_iba, emmodel_ray


def setup_mu(stepsize, bypass_exception=None):
    mu_pos = np.arange(1.0, 0.0, -stepsize)
    if bypass_exception:
        # exclude mu = 1
        mu_pos = mu_pos[1:]
    mu_neg = -mu_pos
    mu = np.concatenate((mu_pos, mu_neg))
    mu = np.array(mu)
    return mu


# Tests to compare with MEMLS IBA, graintype = 2 (small spheres) outputs

@pytest.mark.parametrize("setup_func_pc,memls_ks",
                         [ (0.3e-3, 4.13718676e00),
                           (0.25e-3, 2.58158887e00),
                           (0.2e-3, 1.41304849e00),
                           (0.15e-3, 6.30218291e-01),
                           (0.1e-3, 1.94727497e-01),
                           (0.05e-3, 2.49851702e-02)],
                         indirect=["setup_func_pc"])
def test_ks_pc(setup_func_pc, memls_ks):
    testpack = setup_func_pc
    em = setup_func_em(testpack)
    # Allow 5% error
    assert abs(em.ks(0).meantrace - memls_ks) < tolerance_pc * em.ks(0).meantrace

def test_energy_conservation_exp():
    em = setup_func_em()
    commontest.test_energy_conservation(em, tolerance_pc)


def test_energy_conservation_indep():
    indep_pack = setup_func_indep()
    em = setup_func_em(testpack=indep_pack)
    commontest.test_energy_conservation(em, tolerance_pc)


def test_energy_conservation_shs(setup_func_shs):
    shs_pack = setup_func_shs
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


def test_energy_conservation_shs_active(setup_func_shs):
    shs_pack = setup_func_shs
    em = setup_func_active(testpack=shs_pack)
    commontest.test_energy_conservation(em, tolerance_pc, npol=2)


# def test_energy_conservation_shs_active_but_npol_is_2():
#     shs_pack = setup_func_shs()
#     em = setup_func_active(testpack=shs_pack)
#     commontest.test_energy_conservation(em, tolerance_pc, npol=2)

@pytest.mark.parametrize("bypass_exception", [(None, True)])
def test_iba_vs_rayleigh_m0(setup_func_rayleigh, bypass_exception):
    em_iba, em_ray = setup_func_rayleigh
    mu = setup_mu(1.0 / 64, bypass_exception=bypass_exception)
    assert (
        abs(
            em_iba.ft_even_phase(mu, mu, 0, npol=2) / em_iba.ks(0).meantrace
            - em_ray.ft_even_phase(mu, mu, 0, npol=2) / em_ray.ks(0).meantrace
        )
        < tolerance_pc
    ).all()

def test_iba_vs_rayleigh_active_m1(setup_func_rayleigh):
    em_iba, em_ray = setup_func_rayleigh
    mu = setup_mu(1.0 / 64, bypass_exception=True)
    # Clear cache
    em_iba.cached_mu = None
    assert (
        abs(
            em_iba.ft_even_phase(mu, mu, 1, npol=3)[:, :, 1] / em_iba.ks(0).meantrace
            - em_ray.ft_even_phase(mu, mu, 1, npol=3)[:, :, 1] / em_ray.ks(0).meantrace
        )
        < tolerance_pc
    ).all()


def test_iba_vs_rayleigh_active_m2(setup_func_rayleigh):
    em_iba, em_ray = setup_func_rayleigh
    mu = setup_mu(1.0 / 64, bypass_exception=True)

    def check(i, j):
        print(
            em_iba.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_iba.ks(0).meantrace,
            abs(em_ray.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_ray.ks(0).meantrace),
        )
        assert abs(
            (
                abs(em_iba.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_iba.ks(0).meantrace)
                - abs(em_ray.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_ray.ks(0).meantrace)
            )
            < tolerance_pc
        ).all()

        assert (
            abs(
                em_iba.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_iba.ks(0).meantrace
                - em_ray.ft_even_phase(mu, mu, 2, npol=3)[2, i, j] / em_ray.ks(0).meantrace
            )
            < tolerance_pc
        ).all()

    check(0, 0)
    check(0, 1)
    check(0, 2)
    check(1, 0)
    check(1, 1)
    check(1, 2)
    check(2, 0)
    check(2, 1)
    check(2, 2)


def test_iba_raise_exception_mu_is_1(setup_func_shs):
    shs_pack = setup_func_shs
    em = setup_func_active(testpack=shs_pack)
    bad_mu = np.array([0.2, 1])
    with pytest.raises(SMRTError):
        em.ft_even_phase(bad_mu, bad_mu, 2, npol=3)[:, :, 2]
