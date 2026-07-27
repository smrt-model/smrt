# coding: utf-8

import numpy as np

# local import
from smrt import make_model, make_snowpack, sensor_list

# prepare inputs

length = 2
n_max_stream = 64

nl = length // 2  # // Forces integer division
thickness = np.array([0.1, 0.1] * nl)
thickness[-1] = 100  # last one is semi-infinit
radius = np.array([2e-4] * length)
p_ex = np.array([5e-5] * length)
temperature = np.array([250.0, 250.0] * nl)
density = [200, 400] * nl
stickiness = [0.1, 0.1] * nl

# create the snowpack
snowpack = make_snowpack(
    thickness,
    "sticky_hard_spheres",
    density=density,
    temperature=temperature,
    radius=radius,
    stickiness=stickiness,
)

# create the EM Model
m = make_model("dmrt_shortrange", "dort")

# create the sensor
sensor = sensor_list.amsre("37V")

# run the model
res = m.run(sensor, snowpack)

# outputs
print(res.TbV(), " ", res.TbH())
