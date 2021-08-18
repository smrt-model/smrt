

import pytest
import numpy as np

from ..inputs.make_medium import make_snowpack
from .error import SMRTError
from .interface import Substrate
from .atmosphere import AtmosphereBase


def test_profile():

    sp = make_snowpack([0.1, 0.2, 0.3], "exponential", density=[100, 200, 300], corr_length=200e-6)


    assert np.allclose(sp.z, [0, 0.1, 0.3, 0.6])
    assert np.allclose(sp.bottom_layer_depths, [0.1, 0.3, 0.6])
    assert np.allclose(sp.top_layer_depths, [0.0, 0.1, 0.3])
    assert np.allclose(sp.mid_layer_depths, [0.05, 0.2, 0.45])
    assert np.allclose(sp.profile('density'), [100, 200, 300])


def create_two_snowpacks():
    sp1 = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
    sp2 = make_snowpack([0.5], "exponential", density=400, corr_length=100e-6)

    return sp1, sp2


def test_addition():

    sp1, sp2 = create_two_snowpacks()
    sp = sp1 + sp2

    assert len(sp.layers) == 2
    assert sp.bottom_layer_depths[-1] == 0.6
    assert sp.layers[0].density == 300


def test_layer_addition():

    sp1, sp2 = create_two_snowpacks()

    sp = sp1 + sp2.layers[0]

    assert len(sp.layers) == 2
    assert sp.bottom_layer_depths[-1] == 0.6
    assert sp.layers[0].density == 300

    sp = sp1.layers[0] + sp2

    assert len(sp.layers) == 2
    assert sp.bottom_layer_depths[-1] == 0.6
    assert sp.layers[0].density == 300


def test_inplace_addition():

    sp1, sp2 = create_two_snowpacks()

    sp1 += sp2

    assert len(sp1.layers) == 2
    assert sp1.bottom_layer_depths[-1] == 0.6
    assert sp1.layers[0].density == 300

def test_inplace_layer_addition():

    sp1, sp2 = create_two_snowpacks()

    sp1 += sp2.layers[0]

    assert len(sp1.layers) == 2
    assert sp1.bottom_layer_depths[-1] == 0.6
    assert sp1.layers[0].density == 300


def test_substrate_addition():

    substrate = Substrate()

    sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
    sp += substrate

    assert sp.substrate is substrate

def test_atmosphere_addition():

    atmosphere = AtmosphereBase()  # this one do nothing, but does not matter here.

    sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
    sp = atmosphere + sp

    assert sp.atmosphere is atmosphere


def test_atmosphere_addition_double_snowpack():

    atmosphere = AtmosphereBase()  # this one do nothing, but does not matter here.

    sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
    sp2 = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)

    sp = (atmosphere + sp) + sp2

    assert sp.atmosphere is atmosphere


def test_invalid_addition_atmosphere():

    atmosphere = AtmosphereBase()  # this one do nothing, but does not matter here.

    sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)

    with pytest.raises(SMRTError):
        sp = sp + atmosphere


def test_invalid_addition_atmosphere2():

    atmosphere = AtmosphereBase()  # this one do nothing, but does not matter here.

    sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
    sp2 = atmosphere + make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)

    with pytest.raises(SMRTError):
        sp += sp2


def test_invalid_addition_substrate():

    substrate = Substrate()

    sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)

    with pytest.raises(SMRTError):
        sp = substrate + sp


def test_invalid_addition_substrate2():

    substrate = Substrate()

    sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
    sp2 = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)

    sp += substrate

    with pytest.raises(SMRTError):
        sp += sp2  # the first snowpack can not have a substrate
