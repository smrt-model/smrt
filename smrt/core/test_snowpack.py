

from ..inputs.make_medium import make_snowpack
from .error import SMRTError
from .interface import Substrate
from .atmosphere import AtmosphereBase


def test_addition():

    sp1 = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
    sp2 = make_snowpack([0.5], "exponential", density=400, corr_length=100e-6)

    sp = sp1 + sp2

    assert len(sp.layers) == 2
    assert sp.layer_depths[-1] == 0.6
    assert sp.layers[0].density == 300


def test_inplace_addition():

    sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
    sp += make_snowpack([0.5], "exponential", density=400, corr_length=100e-6)

    assert len(sp.layers) == 2
    assert sp.layer_depths[-1] == 0.6
    assert sp.layers[0].density == 300


def test_substrate_addition():

    substrate = Substrate()

    sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
    sp += substrate

    assert sp.substrate is substrate

def test_atmosphere_addition():

    atmosphere = AtmosphereBase()  # this one do nothing, but does not matter here.

    sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
    sp += atmosphere

    assert sp.atmosphere is atmosphere


def test_atmosphere_addition_double_snowpack():

    atmosphere = AtmosphereBase()  # this one do nothing, but does not matter here.

    sp = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)
    sp2 = make_snowpack([0.1], "exponential", density=300, corr_length=200e-6)

    sp = (sp + atmosphere) + sp2

    assert sp.atmosphere is atmosphere
