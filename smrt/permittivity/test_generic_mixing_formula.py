# coding: utf-8

import numpy as np
import pytest

from ..core.globalconstants import DENSITY_OF_ICE
from .generic_mixing_formula import general_polder_van_santen, polder_van_santen


@pytest.mark.parametrize(
    "inclusion_shape, expected_permittivity",
    [
        (None, 1.52461995825),
        ("random_needles", 1.55052802036),
        ({"spheres": 0.5, "random_needles": 0.5}, 1.53757398931),
    ],
)
def test_pvsl(inclusion_shape, expected_permittivity):
    effective_permittivity = polder_van_santen(frac_volume=(300.0 / DENSITY_OF_ICE), inclusion_shape=inclusion_shape)
    print(effective_permittivity)
    np.testing.assert_allclose(effective_permittivity, expected_permittivity)


def test_general_pvs():
    frac_volume = 300.0 / DENSITY_OF_ICE
    eps = 3 + 0.001j

    eps_eff_g = general_polder_van_santen(frac_volume, eps=eps, depolarization_factors=[1 / 3, 1 / 3, 1 / 3])

    eps_eff = polder_van_santen(frac_volume, eps=eps)

    print(eps_eff, eps_eff_g)

    np.testing.assert_allclose(eps_eff_g, eps_eff)
