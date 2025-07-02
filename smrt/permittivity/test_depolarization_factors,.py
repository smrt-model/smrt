

from .depolarization_factors import depolarization_factors_spheroids

high_tolerance = 1e-8
mid_tolerance = 1e-6
low_tolerance = 0.005

def test_isotropic_default_depolarization_factors():
    depol = depolarization_factors_spheroids()
    assert abs(depol[0] - depol[2]) < high_tolerance


def test_plates_depol():
    depol = depolarization_factors_spheroids(length_ratio=1.5)
    assert depol[0] > depol[2]


def test_hoar_columns_depol():
    depol = depolarization_factors_spheroids(length_ratio=0.5)
    assert depol[0] < depol[2]


def test_depol_approach_to_isotropy_above():
    depol = depolarization_factors_spheroids(length_ratio=1.01)
    assert abs(depol[0] - (1. / 3.)) < low_tolerance


def test_depol_approach_to_isotropy_below():
    depol = depolarization_factors_spheroids(length_ratio=0.99)
    assert abs(depol[0] - (1. / 3.)) < low_tolerance
