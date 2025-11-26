# coding: utf-8

import numpy as np
import pytest

from smrt import make_snow_layer
from smrt.core.sensor import active
from smrt.emmodel import commontest
from smrt.emmodel.symsce_torquato21 import SymSCETK21, derived_SymSCETK21
from smrt.emmodel.test_iba import (
    setup_func_indep,
    setup_func_pc,
    setup_func_sp,
)  # move to a common test file
from smrt.inputs.sensor_list import amsre
from smrt.microstructure_model.sticky_hard_spheres import StickyHardSpheres
from smrt.permittivity.generic_mixing_formula import maxwell_garnett

tolerance = 1e-7
tolerance_pc = 0.001  # 1% error is allowable for differences from MEMLS values. Tests pass at 2%. Some fail at 1%.


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


def setup_func_em(testpack=None):
    if testpack is None:
        testpack = setup_func_sp()
    sensor = amsre("37V")
    emmodel = SymSCETK21(sensor, testpack)
    return emmodel


def setup_func_active(testpack=None):
    if testpack is None:
        testpack = setup_func_sp()
    scatt = active(frequency=10e9, theta_inc=50)
    emmodel = SymSCETK21(scatt, testpack)
    return emmodel


@pytest.mark.parametrize(
    "pc,initial_ks",
    [
        (0.3e-3, 7.4438717),
        (0.25e-3, 4.62265399),
        (0.2e-3, 2.51748175e00),
        (0.15e-3, 1.11772796),
        (0.1e-3, 0.344311),
        (0.05e-3, 0.04413892),
    ],
)
def test_ks(pc, initial_ks):
    testpack = setup_func_pc(pc)
    em = setup_func_em(testpack)
    # Allow 1% error
    print(initial_ks, em.ks(0))
    np.testing.assert_allclose(em.ks(0).meantrace, initial_ks, rtol=tolerance_pc)


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


def test_effective_permittivity_model():
    new_sce = derived_SymSCETK21(effective_permittivity_model=maxwell_garnett)
    layer = setup_func_pc(0.3e-3)
    sensor = amsre("37V")
    new_sce(sensor, layer)
