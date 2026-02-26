# coding: utf-8

import numpy as np

# local import
from smrt import make_model, make_snowpack, sensor_list


def test_mixed_emmodel():
    # prepare inputs
    l = 2

    nl = l // 2  # // Forces integer division
    thickness = np.array([0.1, 0.1] * nl)
    thickness[-1] = 100  # last one is semi-infinit
    radius = np.array([2e-4] * l)
    temperature = np.array([250.0, 250.0] * nl)
    density = [200, 400] * nl
    stickiness = [0.1, 0.1] * nl
    emmodel = ["dmrt_qcacp_shortrange", "iba"] * nl

    # create the snowpack
    snowpack = make_snowpack(
        thickness, "sticky_hard_spheres", density=density, temperature=temperature, radius=radius, stickiness=stickiness
    )

    # create the EM Model
    m = make_model(emmodel, "dort")

    # create the sensor
    radiometer = sensor_list.amsre("37V")

    # run the model
    res = m.run(radiometer, snowpack)

    print(res.TbV(), res.TbH())

    assert (res.TbV() - 204.510189893163) < 1e-4
    assert (res.TbH() - 190.53692754287889) < 1e-4
