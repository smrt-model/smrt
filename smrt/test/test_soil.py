# coding: utf-8

# local import
from smrt import make_snowpack, make_model, sensor, make_soil

#
# Ghi: rapid hack, should be splitted in different functions
#


def prepare_snowpack(substrate):

    density = [300, 300]
    temperature = [245, 245]
    thickness = [0.1, 0.1]
    radius = [200e-6, 400e-6]
    stickiness = [1000, 1000]

    snowpack = make_snowpack(thickness, "sticky_hard_spheres",
                        density=density, temperature=temperature,
                        radius=radius, stickiness=stickiness,
                        substrate=substrate)
    return snowpack


def run_model(snowpack):

    # create the EM Model
    m = make_model("dmrt_qcacp_shortrange", "dort")

    # create the sensor
    radiometer = sensor.passive(37e9, 40)  # test at 40° to avoid the Brewster angle

    # run the model
    res = m.run(radiometer, snowpack)
    return res


def test_soil_wegmuller_dobson85():
    # prepare inputs

    soiltemperature = 270

    clay = 0.3
    sand = 0.4
    drymatter = 1100
    moisture = 0.2
    roughness_rms = 1e-2

    substrate = make_soil("soil_wegmuller", "dobson85", soiltemperature, moisture=moisture, roughness_rms=roughness_rms,
                          clay=clay, sand=sand, drymatter=drymatter)
    snowpack = prepare_snowpack(substrate)

    res = run_model(snowpack)

    print(res.TbV(), res.TbH())
    assert abs(res.TbV() - 262.55457107119486) < 1e-4
    assert abs(res.TbH() - 255.81725907587176) < 1e-4
    # note value from DMRTML Fortran running in the same conditions:
    # H=255.88187817295605 V=262.60345275739024


def test_soil_wegmuller_montpetit2008():
    # prepare inputs

    soiltemperature = 270
    roughness_rms = 1e-2

    substrate = make_soil("soil_wegmuller", "montpetit2008", soiltemperature, roughness_rms=roughness_rms)
    snowpack = prepare_snowpack(substrate)

    res = run_model(snowpack)

    print(res.TbV(), res.TbH())
    assert abs(res.TbV() - 262.4543081568107) < 1e-4
    assert abs(res.TbH() - 255.71089039573724) < 1e-4
    # note value from DMRTML Fortran running in the same conditions:
    # H=255.88187817295605 V=262.60345275739024