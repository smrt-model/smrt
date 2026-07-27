# coding: utf-8


import numpy as np
import numpy.testing as npt
import pytest

# local import
from smrt import make_model, sensor_list
from smrt.inputs.make_soil import make_soil_column


@pytest.fixture
def setup_soil_column():
    thickness = np.array([0.50, 1])  # 50 cm thick layers
    moisture = np.array([0.05, 0.10])
    sand = np.array([0.11, 0.15])
    clay = np.array([0.18, 0.15])
    drymatter = np.array([1300, 1100])
    temperature = np.array([293, 283])  # K

    soil_column = make_soil_column(
        soil_permittivity_model="soil_permittivity_dobson85_peplinski95",
        thickness=thickness,
        temperature=temperature,
        moisture=moisture,
        sand=sand,
        clay=clay,
        dry_matter=drymatter,
    )
    return soil_column


def test_multifresnel_soil(setup_soil_column, atol=1e-2):
    # prepare inputs
    soil = setup_soil_column
    m = make_model("nonscattering", "multifresnel_thermalemission")
    # create the sensor
    radiometer = sensor_list.smos(40)
    # run the model
    res = m.run(radiometer, soil)
    print(res.TbV(), res.TbH())
    # absorption with effective permittivity
    npt.assert_allclose(res.TbV(), 277.66059510071136, atol=atol)
    npt.assert_allclose(res.TbH(), 244.80510230045184, atol=atol)
