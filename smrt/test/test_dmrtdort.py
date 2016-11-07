# coding: utf-8

import numpy as np
from nose.tools import ok_

# local import
from smrt import make_snowpack, make_model, make_snow_layer
from smrt.inputs.sensor_list import amsre

#
# Ghi: rapid hack, should be splitted in different functions
#


def test_dmrt_oneconfig():
    # prepare inputs
    l = 2

    nl = l//2  # // Forces integer division
    thickness = np.array([0.1, 0.1]*nl)
    thickness[-1] = 100  # last one is semi-infinit
    radius = np.array([2e-4]*l)
    temperature = np.array([250.0, 250.0]*nl)
    density = [200, 400]*nl
    stickiness = [0.1, 0.1]*nl

    # create the snowpack
    snowpack = make_snowpack(thickness,
                             "sticky_hard_spheres",
                             density=density,
                             temperature=temperature,
                             radius=radius,
                             stickiness=stickiness)

    # create the EM Model
    m = make_model("dmrt_shortrange", "dort")

    # create the sensor
    radiometer = amsre('37V')

    # run the model
    res = m.run(radiometer, snowpack)

    print(res.TbV(), res.TbH())
    ok_((res.TbV() - 202.34939425929616) < 1e-4)
    ok_((res.TbH() - 187.05199255031036) < 1e-4)
