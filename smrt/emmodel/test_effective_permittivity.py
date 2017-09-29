# coding: utf-8

from nose.tools import raises
from nose.tools import eq_
from nose.tools import ok_
import numpy as np

from smrt.emmodel.effective_permittivity import depolarization_factors, maxwell_garnett, polder_van_santen
from smrt.core.globalconstants import DENSITY_OF_ICE


high_tolerance = 1e-8
mid_tolerance = 1e-6
low_tolerance = 0.005


def test_isotropic_default_depolarization_factors():
    depol = depolarization_factors()
    ok_(abs(depol[0] - depol[2]) < high_tolerance)


def test_plates_depol():
    depol = depolarization_factors(length_ratio=1.5)
    assert depol[0] > depol[2]


def test_hoar_columns_depol():
    depol = depolarization_factors(length_ratio=0.5)
    assert depol[0] < depol[2]


def test_depol_approach_to_isotropy_above():
    depol = depolarization_factors(length_ratio=1.01)
    ok_(abs(depol[0] - (1. / 3.)) < low_tolerance)


def test_depol_approach_to_isotropy_below():
    depol = depolarization_factors(length_ratio=0.99)
    ok_(abs(depol[0] - (1. / 3.)) < low_tolerance)


def test_pvs_real():
    effective_permittivity = polder_van_santen(frac_volume=(300. / DENSITY_OF_ICE))
    ok_(abs(effective_permittivity.real - 1.52441173e+00) < high_tolerance)
