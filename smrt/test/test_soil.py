# coding: utf-8

# local import
import pytest

from smrt import make_model, make_snowpack, make_soil, sensor
from smrt.inputs.make_soil import make_soil_column

#
# Ghi: rapid hack, should be splitted in different functions
#


def prepare_snowpack(substrate):
    density = [300, 300]
    temperature = [245, 245]
    thickness = [0.1, 0.1]
    radius = [200e-6, 400e-6]
    stickiness = [1000, 1000]

    snowpack = make_snowpack(
        thickness,
        "sticky_hard_spheres",
        density=density,
        temperature=temperature,
        radius=radius,
        stickiness=stickiness,
        substrate=substrate,
    )
    return snowpack


def run_model(snowpack):
    # create the EM Model
    m = make_model("dmrt_qcacp_shortrange", "dort")

    # create the sensor
    radiometer = sensor.passive(37e9, 40)  # test at 40° to avoid the Brewster angle

    # run the model
    res = m.run(radiometer, snowpack)
    return res


@pytest.mark.parametrize("soil_permittivity_model", ["dobson85_peplinski95", "dobson85_original"])
def test_soil_wegmuller_dobson85(soil_permittivity_model):
    # prepare inputs

    soiltemperature = 270

    clay = 0.3
    sand = 0.4
    drymatter = 1100
    moisture = 0.2
    roughness_rms = 1e-2

    substrate = make_soil(
        "soil_wegmuller",
        soil_permittivity_model,
        soiltemperature,
        moisture=moisture,
        roughness_rms=roughness_rms,
        clay=clay,
        sand=sand,
        drymatter=drymatter,
    )
    snowpack = prepare_snowpack(substrate)

    res = run_model(snowpack)

    print(res.TbV(), res.TbH())
    if soil_permittivity_model == "dobson85_peplinski95":
        assert abs(res.TbV() - 262.5735899023818) < 1e-4
        assert abs(res.TbH() - 255.85856778263752) < 1e-4
        # note value from DMRTML Fortran running in the same conditions:
        # H=255.88187817295605 V=262.60345275739024
    elif soil_permittivity_model == "dobson85_original":
        assert abs(res.TbV() - 262.56816517455616) < 1e-4
        assert abs(res.TbH() - 255.8528128244208) < 1e-4
    else:
        raise ValueError("Unexpected soil_permittivity_model")


def test_soil_wegmuller_montpetit2008():
    # prepare inputs

    soiltemperature = 270
    roughness_rms = 1e-2

    substrate = make_soil("soil_wegmuller", "montpetit2008", soiltemperature, roughness_rms=roughness_rms)
    snowpack = prepare_snowpack(substrate)

    res = run_model(snowpack)

    print(res.TbV(), res.TbH())
    assert abs(res.TbV() - 262.47365350048574) < 1e-4
    assert abs(res.TbH() - 255.75254543866822) < 1e-4
    # note value from DMRTML Fortran running in the same conditions:
    # H=255.88187817295605 V=262.60345275739024


def test_soil_column():
    # prepare inputs

    soiltemperature = 270

    soil_column = make_soil_column(
        thickness=[1],
        temperature=soiltemperature,
        soil_permittivity_model=None,  # use the default
        moisture=0.2,
        sand=0.4,
        clay=0.3,
        dry_matter=1100,
    )

    m = make_model("nonscattering", "dort")

    # create the sensor
    radiometer = sensor.passive(1.4e9, 40)

    # run the model
    res = m.run(radiometer, soil_column)

    print(res.TbV(), res.TbH())
    assert abs(res.TbV() - 210.77753148744148) < 1e-4
    assert abs(res.TbH() - 159.44393511008025) < 1e-4
    # note value from DMRTML Fortran running in the same conditions:
    # H=255.88187817295605 V=262.60345275739024
