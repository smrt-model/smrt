
import pytest

import numpy as np
from smrt.permittivity.ice import ice_permittivity_maetzler06

from .make_medium import make_snowpack, make_ice_column, make_medium, make_transparent_volume, make_water_body
from ..core.error import SMRTError, SMRTWarning
from ..interface.flat import Flat
from ..interface.transparent import Transparent


def test_make_snowpack():

    sp = make_snowpack(thickness=[1, 2], microstructure_model="exponential", density=[300, 200], corr_length=200e-6)
    assert len(sp.layers) == 2
    assert len(sp.interfaces) == 2
    assert sp.layers[0].thickness == 1
    assert sp.layers[0].density == 300
    assert sp.layers[0].microstructure.corr_length == 200e-6 and sp.layers[1].microstructure.corr_length == 200e-6
    assert sp.bottom_layer_depths[-1] == 3


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
        sp = make_snowpack(thickness=[1], microstructure_model="exponential", density=300, corr_length=200e-6,
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


def test_make_snowpack_volumetric_liquid_water():

    sp = make_snowpack(thickness=[1],
                       microstructure_model="exponential",
                       density=300,
                       corr_length=200e-6
                       )

    assert np.allclose(sp.layers[0].frac_volume, 300 / 916.7)
    assert sp.layers[0].liquid_water == 0

    sp = make_snowpack(thickness=[1],
                       microstructure_model="exponential",
                       density=300,
                       volumetric_liquid_water=0.1,   # 10 % volume
                       corr_length=200e-6
                       )

    assert np.allclose(sp.layers[0].frac_volume, 0.31817388458601503)
    assert np.allclose(sp.layers[0].liquid_water, 0.31429355093084654)


def test_update_volumetric_liquid_water():

    sp = make_snowpack(thickness=[1],
                       microstructure_model="exponential",
                       density=300,
                       corr_length=200e-6
                       )

    assert sp.layers[0].liquid_water == 0

    sp.layers[0].update(volumetric_liquid_water=0.1)   # 10 % volume

    assert np.allclose(sp.layers[0].frac_volume, 0.31817388458601503)
    assert np.allclose(sp.layers[0].liquid_water, 0.31429355093084654)


def test_snow_set_readonly():

    sp = make_snowpack(thickness=[1],
                       microstructure_model="exponential",
                       density=300,
                       volumetric_liquid_water=0.1,   # 10 % volume
                       corr_length=200e-6
                       )

    with pytest.raises(SMRTError):
        sp.layers[0].density = 400

    with pytest.raises(SMRTError):
        sp.layers[0].volumetric_liquid_water = 0.5


def test_empty_snowpack():

    sp = make_snowpack(thickness=[0],
                       microstructure_model="exponential",
                       density=300,
                       corr_length=200e-6
                       )
    # test that the snowpack has one empty layer
    assert len(sp.layers) == 1
    assert sp.layers[0].thickness == 0
    assert sp.layers[0].frac_volume == 0
    assert sp.layers[0].microstructure_model.__name__ == "Homogeneous"


def test_make_transparent_volume():
    sp = make_transparent_volume()

    # test that the snowpack has one empty layer
    assert len(sp.layers) == 1
    assert sp.layers[0].thickness == 0
    assert sp.layers[0].frac_volume == 0
    assert sp.layers[0].microstructure_model.__name__ == "Homogeneous"


def test_make_water_body():
    sp = make_water_body()

    # test that the snowpack has one empty layer
    assert len(sp.layers) == 1
    assert sp.layers[0].thickness > 100
    assert sp.layers[0].frac_volume == 0
    assert sp.layers[0].microstructure_model.__name__ == "Homogeneous"


def test_make_water_body_with_foam():
    sp = make_water_body(foam_frac_volume=0.01)

    # test that the snowpack has one empty layer
    assert len(sp.layers) == 1
    assert sp.layers[0].thickness > 100
    assert sp.layers[0].frac_volume == 0.01
    assert sp.layers[0].microstructure_model.__name__ == "StickyHardSpheres"


@pytest.fixture
def mixing_formula():
    # return a mocked mixing formula
    def mixing_formula(frequency, temperature, density):
        return 1
    return mixing_formula


@pytest.fixture
def default_snowpack_args():
    return dict(thickness=[1],
                microstructure_model="exponential",
                density=300,
                corr_length=200e-6,
                temperature=273)


def test_warning_mixing_formula_in_make_snowpack(mixing_formula, default_snowpack_args):
    with pytest.warns(SMRTWarning):
        make_snowpack(**default_snowpack_args,
                      ice_permittivity_model=mixing_formula
                      )


def test_warning_mixing_formula_in_make_snowpack2(mixing_formula, default_snowpack_args):
    with pytest.warns(SMRTWarning):
        make_snowpack(**default_snowpack_args,
                      background_permittivity_model=mixing_formula
                      )


def test_warning_mixing_formula_in_make_ice_column(mixing_formula, default_snowpack_args):
    with pytest.warns(SMRTWarning):
        make_ice_column("firstyear",
                        **default_snowpack_args,
                        brine_permittivity_model=mixing_formula)


def test_warning_mixing_formula_in_make_ice_column2(mixing_formula, default_snowpack_args):

    with pytest.warns(SMRTWarning):
        make_ice_column("firstyear",
                        **default_snowpack_args,
                        ice_permittivity_model=mixing_formula)


def test_warning_mixing_formula_in_make_ice_column3(mixing_formula, default_snowpack_args):

    with pytest.warns(SMRTWarning):
        make_ice_column("multiyear",
                        **default_snowpack_args,
                        saline_ice_permittivity_model=mixing_formula)


def test_warning_saline_snow(default_snowpack_args):

    with pytest.warns(SMRTWarning):
        make_snowpack(**default_snowpack_args,
                      salinity=0.1)
