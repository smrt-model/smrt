

from nose.tools import raises
from nose.tools import eq_

from .make_medium import make_snowpack
from ..core.error import SMRTError


def test_make_snowpack():

    sp = make_snowpack(thickness=[1, 2], microstructure_model="exponential", density=[300, 200], corr_length=200e-6)
    assert(sp.layers[0].thickness == 1)
    assert (sp.layers[0].density == 300)
    assert (sp.layers[0].microstructure.corr_length == 200e-6 and sp.layers[1].microstructure.corr_length == 200e-6)
    assert(sp.layer_depths[-1] == 3)


@raises(SMRTError)
def test_make_snowpack_with_scalar_thickness():

    sp = make_snowpack(thickness=1, microstructure_model="exponential", density=300, corr_length=200e-6)


@raises(SMRTError)
def test_make_snowpack_array_size():

    # should raise an exception because density is len 1 whereas thickness is len 2
    sp = make_snowpack(thickness=[1, 2], microstructure_model="exponential", density=[300], corr_length=200e-6)

