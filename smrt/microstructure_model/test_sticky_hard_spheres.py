# coding: utf-8

import numpy as np
from .sticky_hard_spheres import StickyHardSpheres


def test_constructor():

    shs = StickyHardSpheres({'radius': 0.001, 'frac_volume': 0.3})


def test_autocorrelation():

    shs = StickyHardSpheres({'radius': 0.001, 'frac_volume': 0.3})

    r = np.arange(0, 1e-3, 1e-4)
    shs.autocorrelation_function(r)