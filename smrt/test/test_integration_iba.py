# coding: utf-8

import numpy as np
from nose.tools import ok_

# local import
from smrt import make_snowpack, make_model, sensor_list

#
# Ghi: rapid hack, should be splitted in different functions
#

def test_iba_oneconfig():

    # prepare inputs

    l = 2
    n_max_stream = 64

    nl = l//2  # // Forces integer division
    thickness = np.array([0.1, 0.1]*nl)
    thickness[-1] = 100  # last one is semi-infinit
    p_ex = np.array([5e-5]*l)
    temperature = np.array([250.0, 250.0]*nl)
    density = [200, 400]*nl

    # create the snowpack
    snowpack = make_snowpack(thickness=thickness,
                             microstructure_model="exponential",
                             density=density,
                             temperature=temperature,
                             corr_length=p_ex)

    # create the snowpack
    m = make_model("iba", "dort")

    # create the sensor
    radiometer = sensor_list.amsre('37V')

    # run the model
    res = m.run(radiometer, snowpack)

    print(res.TbV(), res.TbH())
    #absorption with effective permittivity
    # ok_(abs(res.TbV() - 248.08794944809972) < 1e-4)
    # ok_(abs(res.TbH() - 237.3056263719142) < 1e-4)

    ok_(abs(res.TbV() - 248.08744066791073) < 1e-4)
    ok_(abs(res.TbH() - 237.30720491883298) < 1e-4)

