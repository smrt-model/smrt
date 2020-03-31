
import pandas as pd
import numpy as np

from nose.tools import raises

from .make_medium import make_snowpack, make_ice_column, make_medium
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


def test_make_lake_ice():

    sp = make_ice_column("fresh", thickness=[1], microstructure_model="exponential", density=[300], corr_length=200e-6, temperature=273)
    assert(sp.layers[0].thickness == 1)
    assert (sp.layers[0].density == 300)
    assert (sp.layers[0].microstructure.corr_length == 200e-6)


def test_make_medium():

    sp_dict = {
        'thickness': [0.1, 1],
        'density': [200, 300],
        'microstructure_model': 'sticky_hard_spheres',
        'radius': [100e-6, 100e-6],
        'temperature': 273
    }

    sp = make_medium(sp_dict)

    assert np.allclose(sp.layer_thicknesses, sp_dict['thickness'])
    assert np.allclose([l.temperature for l in sp.layers], sp_dict['temperature'])
    assert np.allclose([l.microstructure.radius for l in sp.layers], sp_dict['radius'])
