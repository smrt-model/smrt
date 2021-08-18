# coding: utf-8

import numpy as np

from .generic_mixing_formula import depolarization_factors, maxwell_garnett, polder_van_santen
from ..core.globalconstants import DENSITY_OF_ICE


high_tolerance = 1e-8
mid_tolerance = 1e-6
low_tolerance = 0.005


def test_isotropic_default_depolarization_factors():
    depol = depolarization_factors()
    assert abs(depol[0] - depol[2]) < high_tolerance


def test_plates_depol():
    depol = depolarization_factors(length_ratio=1.5)
    assert depol[0] > depol[2]


def test_hoar_columns_depol():
    depol = depolarization_factors(length_ratio=0.5)
    assert depol[0] < depol[2]


def test_depol_approach_to_isotropy_above():
    depol = depolarization_factors(length_ratio=1.01)
    assert abs(depol[0] - (1. / 3.)) < low_tolerance


def test_depol_approach_to_isotropy_below():
    depol = depolarization_factors(length_ratio=0.99)
    assert abs(depol[0] - (1. / 3.)) < low_tolerance


def test_pvsl_spheres():
    effective_permittivity = polder_van_santen(frac_volume=(300. / DENSITY_OF_ICE))
    print(effective_permittivity)
    assert abs(effective_permittivity - 1.52461995825) < high_tolerance

def test_pvsl_needles():
    effective_permittivity = polder_van_santen(frac_volume=(300. / DENSITY_OF_ICE), inclusion_shape="random_needles")
    print(effective_permittivity)
    assert abs(effective_permittivity - 1.55052802036) < high_tolerance

def test_pvsl_mix_spheres_needles():
    effective_permittivity = polder_van_santen(frac_volume=(300. / DENSITY_OF_ICE), inclusion_shape={"spheres": 0.5, "random_needles": 0.5})
    print(effective_permittivity)
    assert abs(effective_permittivity - 1.53757398931) < high_tolerance
