# coding: utf-8

import numpy as np
import pytest

# local import
from smrt import PSU, make_model, sensor_list
from smrt.inputs.make_medium import bulk_ice_density, make_ice_column

# test if this configuration gives values as originally produced by examples/iba_sea_ice.py
# same structure as test_integration_iba.py


@pytest.fixture
def setup_seaice():
    l = 9  # 9 ice layers
    n_max_stream = 64

    thickness = np.array([1.5 / l] * l)  # ice is 1.5m thick
    # temperature gradient in the ice from -20 deg C at top to freezing temperature of water at bottom (-1.8 deg C)
    temperature = np.linspace(273.15 - 20.0, 273.15 - 1.8, l)
    # salinity profile ranging from salinity=2 at the top to salinity=10 at the bottom of the ice
    salinity = np.linspace(2.0, 10.0, l) * PSU

    return l, n_max_stream, thickness, temperature, salinity


@pytest.mark.parametrize(
    "ice_type,porosity,p_ex_value,results",
    [
        ("firstyear", 0, 500e-6, [256.0170296269674, 228.4566040823167]),
        ("multiyear", 0.08, 1000e-6, [257.57209000420636, 232.01555447145563]),
    ],
)
def test_oneconfig_for_sea_ice(setup_seaice, ice_type, porosity, p_ex_value, results):
    # prepare inputs
    l, n_max_stream, thickness, temperature, salinity = setup_seaice
    p_ex = np.array([p_ex_value] * l)  # correlation length

    # create an ice column with assumption of spherical brine inclusions (brine_inclusion_shape="spheres"):
    ice_column = make_ice_column(
        ice_type=ice_type,
        thickness=thickness,
        temperature=temperature,
        microstructure_model="exponential",
        brine_inclusion_shape="spheres",  # inclusion_shape can be "spheres" or "random_needles", or "mix_spheres_needles"
        salinity=salinity,  # either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice
        porosity=porosity,
        corr_length=p_ex,
        add_water_substrate="ocean",
    )

    # create the sensor
    sensor = sensor_list.passive(1.4e9, 40.0)
    n_max_stream = 128
    m = make_model("iba", "dort", rtsolver_options={"n_max_stream": n_max_stream})

    # run the model
    res = m.run(sensor, ice_column)

    print(res.TbV(), res.TbH())
    # absorption with effective permittivity
    assert abs(res.TbV() - results[0]) < 1e-4
    assert abs(res.TbH() - results[1]) < 1e-4


def test_equivalence_porosity_density(setup_seaice):
    l, n_max_stream, thickness, temperature, salinity = setup_seaice
    p_ex = np.array([1.0e-3] * (l))  # correlation length
    ice_type = "multiyear"  # first-year (FY) or multi-year (MY) sea ice
    porosity = 0.08  # ice porosity, in fraction

    ice_column1 = make_ice_column(
        ice_type,
        thickness=thickness,
        temperature=temperature,
        microstructure_model="exponential",
        brine_inclusion_shape="spheres",
        # brine_inclusion_shape can be "spheres", "random_needles" or "mix" (a mix of spheres and needles)
        salinity=salinity,
        # either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice
        porosity=porosity,
        # either density or 'porosity' should be set for sea ice. If porosity is given, density is calculated in the model. If none is given, ice is treated as having a porosity of 0% (no air inclusions)
        #                            density = density,
        corr_length=p_ex,
        add_water_substrate="ocean",  # see comment below
    )

    # Same, but giving the density instead:
    density = [bulk_ice_density(temp, salt, porosity) for temp, salt in zip(temperature, salinity)]

    ice_column2 = make_ice_column(
        ice_type,
        thickness=thickness,
        temperature=temperature,
        microstructure_model="exponential",
        brine_inclusion_shape="spheres",
        # brine_inclusion_shape can be "spheres", "random_needles" or "mix" (a mix of spheres and needles)
        salinity=salinity,
        # either 'salinity' or 'brine_volume_fraction' should be given for sea ice; if salinity is given, brine volume fraction is calculated in the model; if none is given, ice is treated as fresh water ice
        #                            porosity = porosity, # either density or 'porosity' should be set for sea ice. If porosity is given, density is calculated in the model. If none is given, ice is treated as having a porosity of 0% (no air inclusions)
        density=density,
        corr_length=p_ex,
        add_water_substrate="ocean",  # see comment below
    )

    sensor = sensor_list.passive(1.4e9, 40.0)

    n_max_stream = 128  # TB calculation is more accurate if number of streams is increased (currently: default = 32);
    # needs to be increased when using > 1 snow layer on top of sea ice!
    m = make_model("iba", "dort", rtsolver_options={"n_max_stream": n_max_stream})

    # run the model for sea ice with porosity / density

    res1 = m.run(sensor, ice_column1)
    res2 = m.run(sensor, ice_column2)
    print(res1.TbV(), res1.TbH())
    print(res2.TbV(), res2.TbH())

    # The two should be similar
    assert abs(res1.TbV() - res2.TbV()) < 1e-4
    assert abs(res1.TbH() - res2.TbH()) < 1e-4
