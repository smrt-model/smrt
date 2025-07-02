# coding: utf-8

import numpy as np

from .generic_mixing_formula import maxwell_garnett, polder_van_santen, general_polder_van_santen
from ..core.globalconstants import DENSITY_OF_ICE


def test_pvsl_spheres():
    effective_permittivity = polder_van_santen(frac_volume=(300.0 / DENSITY_OF_ICE))
    print(effective_permittivity)
    np.testing.assert_allclose(effective_permittivity, 1.52461995825)


def test_pvsl_needles():
    effective_permittivity = polder_van_santen(frac_volume=(300.0 / DENSITY_OF_ICE), inclusion_shape="random_needles")
    print(effective_permittivity)
    np.testing.assert_allclose(effective_permittivity, 1.55052802036)


def test_pvsl_mix_spheres_needles():
    effective_permittivity = polder_van_santen(
        frac_volume=(300.0 / DENSITY_OF_ICE), inclusion_shape={"spheres": 0.5, "random_needles": 0.5}
    )
    print(effective_permittivity)
    np.testing.assert_allclose(effective_permittivity, 1.53757398931)


def test_general_pvs():
    frac_volume = 300.0 / DENSITY_OF_ICE
    eps = 3 + 0.001j

    eps_eff_g = general_polder_van_santen(frac_volume, eps=eps, depolarization_factors=[1 / 3, 1 / 3, 1 / 3])

    eps_eff = polder_van_santen(frac_volume, eps=eps)

    print(eps_eff, eps_eff_g)

    np.testing.assert_allclose(eps_eff_g, eps_eff)
