
import pytest

import numpy as np

from .make_medium import make_snowpack, make_ice_column, make_medium
from ..core.error import SMRTError
from ..interface.flat import Flat
from ..interface.transparent import Transparent


def test_make_snowpack():

    sp = make_snowpack(thickness=[1, 2], microstructure_model="exponential", density=[300, 200], corr_length=200e-6)
    assert len(sp.layers) == 2
    assert len(sp.interfaces) == 2
    assert sp.layers[0].thickness == 1
    assert sp.layers[0].density == 300
    assert sp.layers[0].microstructure.corr_length == 200e-6 and sp.layers[1].microstructure.corr_length == 200e-6
    assert sp.layer_depths[-1] == 3


def test_make_snowpack_surface_interface():

    sp = make_snowpack(thickness=[1, 2], microstructure_model="exponential", density=[300, 200], corr_length=200e-6, surface="transparent")
    assert isinstance(sp.interfaces[0], Transparent)
    assert isinstance(sp.interfaces[1], Flat)


def test_make_snowpack_interface():

    interfaces = [Transparent, Flat]
    sp = make_snowpack(thickness=[1, 2], microstructure_model="exponential", density=[300, 200], corr_length=200e-6, interface=interfaces)
    assert isinstance(sp.interfaces[0], Transparent)
    assert isinstance(sp.interfaces[1], Flat)


def test_make_snowpack_surface_and_list_interface():

    with pytest.raises(SMRTError):
        sp = make_snowpack(thickness=1, microstructure_model="exponential", density=300, corr_length=200e-6,
                           interfaces=[Transparent, Flat], surface=Flat)


def test_make_snowpack_with_scalar_thickness():
    with pytest.raises(SMRTError):
        sp = make_snowpack(thickness=1, microstructure_model="exponential", density=300, corr_length=200e-6)


def test_make_snowpack_array_size():

    # should raise an exception because density is len 1 whereas thickness is len 2
    with pytest.raises(SMRTError):
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
    assert np.allclose([lay.temperature for lay in sp.layers], sp_dict['temperature'])
    assert np.allclose([lay.microstructure.radius for lay in sp.layers], sp_dict['radius'])
