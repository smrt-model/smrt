# coding: utf-8

import pytest

import numpy as np

from smrt.emmodel.symsce_torquato21 import SymSCETK21, derived_SymSCETK21
from smrt.core.sensor import active
from smrt.inputs.sensor_list import amsre
from smrt.emmodel import commontest
from smrt.permittivity.generic_mixing_formula import maxwell_garnett


tolerance = 1e-7
tolerance_pc = 0.001  # 1% error is allowable for differences from MEMLS values. Tests pass at 2%. Some fail at 1%.

from smrt.emmodel.test_iba import setup_func_sp, setup_func_indep, setup_func_shs, setup_func_pc, setup_mu # move to a common test file


def setup_func_em(testpack=None):
    if testpack is None:
        testpack = setup_func_sp()
    sensor = amsre('37V')
    emmodel = SymSCETK21(sensor, testpack)
    return emmodel


def setup_func_active(testpack=None):
    if testpack is None:
        testpack = setup_func_sp()
    scatt = active(frequency=10e9, theta_inc=50)
    emmodel = SymSCETK21(scatt, testpack)
    return emmodel


def test_ks_pc_is_0p3_mm():
    testpack = setup_func_pc(0.3e-3)
    em = setup_func_em(testpack)
    # Allow 1% error
    initial_ks = 7.4438717
    print(initial_ks, em.ks(0))
    np.testing.assert_allclose(em.ks(0).meantrace, initial_ks, rtol=tolerance_pc)


def test_ks_pc_is_0p25_mm():
    testpack = setup_func_pc(0.25e-3)
    em = setup_func_em(testpack)
    # Allow 1% error
    initial_ks = 4.62265399
    print(initial_ks, em.ks(0))
    np.testing.assert_allclose(em.ks(0).meantrace, initial_ks, rtol=tolerance_pc)


def test_ks_pc_is_0p2_mm():
    testpack = setup_func_pc(0.2e-3)
    em = setup_func_em(testpack)
    # Allow 1% error
    initial_ks = 1.41304849e+00
    print(initial_ks, em.ks(0))
    np.testing.assert_allclose(em.ks(0).meantrace, initial_ks, rtol=tolerance_pc)



def test_ks_pc_is_0p15_mm():
    testpack = setup_func_pc(0.15e-3)
    em = setup_func_em(testpack)
    # Allow 1% error
    initial_ks = 1.11772796
    print(initial_ks, em.ks(0))
    np.testing.assert_allclose(em.ks(0).meantrace, initial_ks, rtol=tolerance_pc)


def test_ks_pc_is_0p1_mm():
    testpack = setup_func_pc(0.1e-3)
    em = setup_func_em(testpack)
    # Allow 1% error
    initial_ks = 0.344311
    print(initial_ks, em.ks(0))
    np.testing.assert_allclose(em.ks(0).meantrace, initial_ks, rtol=tolerance_pc)


def test_ks_pc_is_0p2_mm():
    testpack = setup_func_pc(0.05e-3)
    em = setup_func_em(testpack)
    # Allow 1% error
    initial_ks = 0.04413892
    print(initial_ks, em.ks(0))
    np.testing.assert_allclose(em.ks(0).meantrace, initial_ks, rtol=tolerance_pc)


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

def test_effective_permittivity_model():

    new_sce = derived_SymSCETK21(effective_permittivity_model=maxwell_garnett)
    layer = setup_func_pc(0.3e-3)
    sensor = amsre('37V')
    new_sce(sensor, layer)

