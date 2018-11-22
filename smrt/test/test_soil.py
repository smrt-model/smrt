# coding: utf-8

from nose.tools import ok_

# local import
from smrt import make_snowpack, make_model, sensor, make_soil

#
# Ghi: rapid hack, should be splitted in different functions
#


def test_dmrt_with_soil():
    # prepare inputs

    density = [300, 300]
    temperature = [245, 245]
    thickness = [0.1, 0.1]
    radius = [200e-6, 400e-6]
    stickiness = [1000, 1000]
    soiltemperature = 270

    clay = 0.3
    sand = 0.4
    drymatter = 1100
    moisture = 0.2
    roughness_rms = 1e-2

    substrate = make_soil("soil_wegmuller", "dobson85", soiltemperature, moisture=moisture, roughness_rms=roughness_rms,
                          clay=clay, sand=sand, drymatter=drymatter)

    snowpack = make_snowpack(thickness, "sticky_hard_spheres",
                        density=density, temperature=temperature, radius=radius, stickiness=stickiness,
                        substrate=substrate)

    # create the EM Model
    m = make_model("dmrt_qcacp_shortrange", "dort")

    # create the sensor
    radiometer = sensor.passive(37e9, 40)  # test at 40° to avoid the Brewster angle

    # run the model
    res = m.run(radiometer, snowpack)

    print(res.TbV(), res.TbH())
    #ok_((res.TbV() - 262.6214674671272) < 1e-4)
    ok_((res.TbV() - 262.62154074526325) < 1e-4)
    #ok_((res.TbH() - 255.88791903746) < 1e-4)
    ok_((res.TbH() - 255.88831382514428) < 1e-4)
    # note value from DMRTML Fortran running in the same conditions:
    # H=255.88187817295605 V=262.60345275739024